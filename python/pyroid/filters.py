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

import importlib
import sys
import threading
import unicodedata
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any, NamedTuple

from pyroid import _versions
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

_versions.check(__name__, "datatrove", f"datatrove=={DATATROVE_VERSION}")

#: datatrove's default Gopher stop words.
STOP_WORDS = ("the", "be", "to", "of", "and", "that", "have", "with")


class GopherQuality(NamedTuple):
    """``GopherQualityFilter``'s parameters and defaults.

    A falsy threshold (``None``, ``0``) disables its check, as in datatrove.
    ``stop_words=None`` means :data:`STOP_WORDS`.
    """

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
    """``GopherRepetitionFilter``'s parameters and defaults.

    A falsy fraction disables its check. ``top_n_grams`` and ``dup_n_grams``
    are ``(n, max_fraction)`` pairs, ``n >= 1``.
    """

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
    """``C4QualityFilter``'s parameters and defaults.

    ``-1`` disables ``min_num_sentences``, ``min_words_per_line`` or
    ``max_word_length``. ``split_paragraph=False`` filters sentences instead
    of lines (and joins the kept ones with spaces).
    """

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
    """``FineWebQualityFilter``'s parameters and defaults.

    ``stop_chars=None`` means datatrove's ``TERMINAL_PUNCTUATION``.
    """

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
    """How datatrove 0.10.1 tokenizes one language code (an entry of :func:`languages`)."""

    tokenizer: str
    arg: str
    native: bool


_languages_cache: dict[str, dict[str, Language]] = {}


def languages() -> dict[str, Language]:
    """Every language code datatrove 0.10.1 knows, with its tokenizer.

    The keys are what ``language=`` accepts: datatrove's ``tokenizer_assignment.csv``
    as ``load_word_tokenizer`` reads it (``"deu_Latn"``, ``"deu"``, ``"de"``, ...).
    ``pyroid.dedup.languages`` is the same function.

    Example::

        >>> pyroid.filters.languages()["cmn_Hani"]
        Language(tokenizer='SpaCyTokenizer', arg='zh', native=False)

    Returns:
        ``{code: Language(tokenizer, arg, native)}``, sorted by code.
    """
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


#: What makes a tokenizer native when it is installed (pyroid reads its data at run time).
_DATA_HINTS: dict[tuple[str, str], str] = {
    ("SpaCyTokenizer", "ja"): (
        " Or install sudachidict-core (20261015 or newer), and pyroid tokenizes it natively."
    ),
    ("ThaiTokenizer", ""): " Or install pythainlp (5.4.0), and pyroid tokenizes it natively.",
}
_NATIVE_OK: set[tuple[str, str]] = set()


def _native_language(code: str, math: str = "the filter math still runs in Rust") -> None:
    """Raise unless pyroid tokenizes ``code`` natively (datatrove's ValueError if unknown)."""
    version = _unicode_version()
    if (code, version) in _NATIVE_OK:
        return
    cls, arg, native = _backend(code)
    if not native:
        # Native once a SudachiDict dictionary, or PyThaiNLP, is installed.
        dictionary = _DATA_HINTS.get((cls, arg), "")
        raise NotImplementedError(
            f"pyroid has no native word tokenizer for language {code!r}: datatrove tokenizes it "
            f"with {_describe(cls, arg)}. Pass python_tokenizer=True to tokenize it with "
            "datatrove's own tokenizer in Python, on the calling thread "
            f"(needs `{_pip(cls, arg)}`); "
            f"{math}.{dictionary}"
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
    _DT_MODULES[module] = mod
    return mod


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
    """``GopherQualityFilter``: True for each document datatrove keeps.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column
            (``string``, ``large_string`` or ``string_view``).
        config: A :class:`GopherQuality` (or a datatrove
            ``GopherQualityFilter``); keyword arguments override its fields.
        threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at most
            ``n`` tasks at once; ``1`` runs on the calling thread. On the
            Python-tokenizer path it applies to the Rust part only.
        python_tokenizer: Tokenize with datatrove's own tokenizer for
            ``language``, in Python on the calling thread, and run the filter
            math in Rust on its words (see "Words and languages" above). Needed
            for the languages pyroid does not tokenize natively; emits
            :class:`pyroid.FallbackWarning` once per function and language.
        **kwargs: ``GopherQualityFilter``'s parameters (``language``: a
            datatrove language code or a word tokenizer object).

    Returns:
        bool, ``list[bool]``, ``tuple[bool, ...]``, or a :class:`pyroid.Column`
        of bools (nulls stay null), matching ``texts``.

    Raises:
        pyroid.InputError: ``texts`` is not one of the accepted kinds.
        TypeError: A parameter has the wrong type.
        ValueError: datatrove has no tokenizer for ``language`` (datatrove's
            message), or a document of 2**30 characters or more (spaCy's
            limit; 2**32 - 64 for Thai).
        NotImplementedError: pyroid does not tokenize ``language`` natively
            and ``python_tokenizer`` is not set (the message names
            datatrove's tokenizer and what to install).
        pyroid.PyroidError: The Python-tokenizer path needs datatrove or the
            tokenizer's package, and it is not installed.
        ZeroDivisionError: As datatrove, e.g. ``min_doc_words=None`` and a
            document without words.
        UnicodeEncodeError: A str contains a lone surrogate.
        UnicodeDecodeError: An Arrow row is not valid UTF-8.
    """
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
    """``GopherQualityFilter``: datatrove's drop reason, ``None`` when kept.

    Reasons: ``gopher_short_doc``, ``gopher_long_doc``,
    ``gopher_below_avg_threshold``, ``gopher_above_avg_threshold``,
    ``gopher_too_many_hashes``, ``gopher_too_many_ellipsis``,
    ``gopher_too_many_bullets``, ``gopher_too_many_end_ellipsis``,
    ``gopher_below_alpha_threshold``, ``gopher_enough_stop_words``.
    Arguments and errors as for :func:`gopher_quality`; Arrow input gives a
    ``string`` :class:`pyroid.Column` (null when kept or null).
    """
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
    """``GopherRepetitionFilter``: True for each document datatrove keeps.

    Arguments, returns and errors as for :func:`gopher_quality`, with
    :class:`GopherRepetition` / ``GopherRepetitionFilter`` parameters.
    """
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
    """``GopherRepetitionFilter``: datatrove's drop reason, ``None`` when kept.

    Reasons: ``empty``, ``dup_para_frac``, ``dup_para_char_frac``,
    ``dup_line_frac``, ``dup_line_char_frac``, ``top_{n}_gram``,
    ``duplicated_{n}_n_grams``.
    """
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
    """``C4QualityFilter``'s document decision: True for each document kept.

    Arguments, returns and errors as for :func:`gopher_quality`, with
    :class:`C4Quality` / ``C4QualityFilter`` parameters. The text after line
    removal is :func:`c4_lines`.
    """
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
    """``C4QualityFilter``: datatrove's drop reason, ``None`` when kept.

    Reasons: ``lorem_ipsum``, ``curly_bracket``, ``too_few_sentences``.
    """
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
    """``C4QualityFilter``'s line removal: ``doc.text`` after the filter.

    For a kept document, the lines (or sentences) C4 keeps, stripped and
    re-joined, citations removed; for a dropped one, the input (datatrove
    leaves it untouched). Unchanged documents are the original objects;
    Arrow nulls stay null.

    Returns:
        str, list, tuple, or a :class:`pyroid.Column` of strings
        (``large_string`` for ``large_string`` input), matching ``texts``.
    """
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
    """``FineWebQualityFilter``: True for each document datatrove keeps.

    Arguments, returns and errors as for :func:`gopher_quality`, with
    :class:`FineWebQuality` / ``FineWebQualityFilter`` parameters.
    """
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
    """``FineWebQualityFilter``: datatrove's drop reason, ``None`` when kept.

    Reasons: ``empty``, ``line_punct_ratio``, ``short_line_ratio``,
    ``char_dup_ratio``, ``list_ratio``.
    """
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
    """Run several filters in order, in one pass over each document.

    The same as running the datatrove filters one after another in a
    pipeline: a document dropped by one filter is not seen by the next, and
    a :class:`C4Quality` step's rewritten text is what later steps filter.
    Steps that see the same text tokenize it once.

    Args:
        texts: As for :func:`gopher_quality`.
        filters: Config objects, or datatrove filter instances (or one of either)
            (``GopherQualityFilter`` etc.) whose settings are read. Each has its
            own ``language``.
        threads: As for :func:`gopher_quality`.
        python_tokenizer: As for :func:`gopher_quality`, for every step.
            Steps are then run in datatrove's order: consecutive steps with the
            same tokenizer share one tokenization per document, and a C4 step
            runs as datatrove's ``C4QualityFilter``, whose rewritten text the
            later steps tokenize again.

    Returns:
        ``FilterResult(keep, reason, text)``, each in the caller's kind.

    Raises the same exceptions as :func:`gopher_quality`.
    """
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


# ======================================================================================
# More datatrove 0.10.1 steps: one section each (each extends __all__).
# ======================================================================================

# ---- PIIFormatter ----
__all__ += ["PII", "PII_EMAIL_REPLACEMENT", "PII_IP_REPLACEMENT", "PIIReplacer", "pii_replace"]

#: ``PIIFormatter``'s default ``email_replacement``.
PII_EMAIL_REPLACEMENT = ("email@example.com", "firstname.lastname@example.org")
#: ``PIIFormatter``'s default ``ip_replacement``.
PII_IP_REPLACEMENT = (
    "22.214.171.124",
    "126.96.36.199",
    "188.8.131.52",
    "184.108.40.206",
    "220.127.116.11",
    "18.104.22.168",
)


class PII(NamedTuple):
    """``PIIFormatter``'s parameters and defaults.

    Attributes:
        remove_emails: Replace email addresses.
        remove_ips: Replace IPv4 addresses.
        only_remove_public_ips: Replace only addresses ``ipaddress`` calls
            ``is_global`` (the running interpreter's tables, as datatrove).
        email_replacement: Replacements, used in turn (a str is one).
        ip_replacement: The same for IP addresses.
    """

    remove_emails: bool = True
    remove_ips: bool = True
    only_remove_public_ips: bool = True
    email_replacement: tuple[str, ...] | str = PII_EMAIL_REPLACEMENT
    ip_replacement: tuple[str, ...] | str = PII_IP_REPLACEMENT


def _replacements(name: str, v: object) -> list[str]:
    # datatrove's PIIReplacer: a tuple as is, a str as a 1-tuple, any other iterable as a tuple.
    items: Iterable[object] = (v,) if isinstance(v, str) else _iterable(name, v)
    out = []
    for x in items:
        if not isinstance(x, str):
            raise TypeError(f"{name} items must be str, not {type(x).__name__}")
        out.append(x)
    return out


def _iterable(name: str, v: object) -> Iterable[object]:
    if not isinstance(v, Iterable):
        raise TypeError(f"{name} must be a str or an iterable of str, not {type(v).__name__}")
    return v


def _public_ip(addr: str) -> bool:
    """Datatrove's ``public_ip_validator`` (only called on addresses that parse)."""
    import ipaddress

    try:
        return ipaddress.ip_address(addr).is_global
    except ValueError:
        return False


class PIIReplacer:
    """datatrove's ``PIIFormatter``, keeping its state across calls.

    Each call rewrites a batch of documents exactly as ``PIIFormatter.format``
    would, one after another: every match of datatrove's email regex becomes
    the next email replacement, then, in that text, every match of its IPv4
    regex that ``ipaddress.ip_address`` accepts (and, by default, calls
    ``is_global``) becomes the next IP replacement. "Next" cycles through each
    replacement tuple and carries over from one document, and one call, to the
    next, so feeding a corpus batch by batch gives datatrove's output.

    The matches are found and replaced in Rust, in parallel; the calling
    thread numbers the replacements and asks ``ipaddress`` about each distinct
    address once.

    Attributes:
        config: The :class:`PII` settings.
        email_index: Index of the next email replacement (datatrove's
            ``emails_replacer._replace_i``); assignable.
        ip_index: The same for IP replacements (``ip_replacer._replace_i``).
    """

    __slots__ = ("_native", "_validator", "config", "email_index", "ip_index")

    def __init__(self, config: PII | None = None, /, **kwargs: Any) -> None:
        """A formatter whose counters start at 0.

        Args:
            config: A :class:`PII`; keyword arguments override its fields.
            **kwargs: ``PIIFormatter``'s parameters.

        Raises:
            TypeError: A replacement is not a str.
        """
        cfg = _config(PII, config, kwargs)
        self.config: PII = cfg
        self._native = _native._PII(
            bool(cfg.remove_emails),
            bool(cfg.remove_ips),
            _replacements("email_replacement", cfg.email_replacement),
            _replacements("ip_replacement", cfg.ip_replacement),
            _unicode_version(),
        )
        self._validator = _public_ip if cfg.only_remove_public_ips else None
        self.email_index = 0
        self.ip_index = 0

    def __call__(self, texts: Any, /, *, threads: Any = None) -> Any:
        """Rewrite each document; unchanged documents come back as the same objects.

        Args:
            texts: A str, a list or tuple of str, or an Arrow string array/column.
            threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at most
                ``n`` tasks at once; ``1`` runs on the calling thread.

        Returns:
            The texts in the caller's kind (a :class:`pyroid.Column` of strings for
            Arrow input; nulls stay null).

        Raises:
            pyroid.InputError: ``texts`` is not one of the accepted kinds.
            IndexError: A match needs a replacement and the tuple is empty (as datatrove).
            UnicodeEncodeError: A str contains a lone surrogate.
            UnicodeDecodeError: An Arrow row is not valid UTF-8.
        """
        out, self.email_index, self.ip_index = self._native.run(
            texts, self.email_index, self.ip_index, self._validator, threads=threads
        )
        return out

    def __repr__(self) -> str:
        return (
            f"PIIReplacer({self.config!r}, email_index={self.email_index}, "
            f"ip_index={self.ip_index})"
        )


def pii_replace(texts: Any, config: Any = None, /, *, threads: Any = None, **kwargs: Any) -> Any:
    """``PIIFormatter``: each document's text with emails and IP addresses replaced.

    A fresh formatter per call, so the replacement counters start at 0: one
    call over a whole corpus gives datatrove's output. To process a corpus in
    batches, use one :class:`PIIReplacer`.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        config: A :class:`PII`; keyword arguments override its fields.
        threads: As for :meth:`PIIReplacer.__call__`.
        **kwargs: ``PIIFormatter``'s parameters.

    Returns:
        As :meth:`PIIReplacer.__call__`.
    """
    return PIIReplacer(config, **kwargs)(texts, threads=threads)


# ---- end PIIFormatter ----

# ---- C4BadWordsFilter ----
__all__ += [
    "C4_BADWORDS_LANGUAGES",
    "BadWordsFilter",
    "BadWordsResult",
    "C4BadWords",
    "c4_badwords",
    "c4_badwords_reasons",
    "load_c4_badwords",
]

#: The languages datatrove 0.10.1 has a bad-word list for (``_BADWORDS_LANGS``).
C4_BADWORDS_LANGUAGES = (
    "ar", "cs", "da", "de", "en", "eo", "es", "fa", "fi", "fil", "fr", "fr-CA-u-sd-caqc", "hi",
    "hu", "it", "ja", "kab", "ko", "nl", "no", "pl", "pt", "ru", "sv", "th", "tlh", "tr", "zh",
)  # fmt: skip

# The LDNOOBW "List of Dirty, Naughty, Obscene, and Otherwise Bad Words" (CC BY 4.0,
# https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words), at the commits
# datatrove pins; fetched at run time like datatrove does, never shipped with pyroid.
_LDNOOBW = "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/"
_BADWORDS_COMMIT = "5faf2ba42d7b1c0977169ec3611df25a3c08eb13"
_EN_BADWORDS_COMMIT = "25e679f03d96baa721cde20db9944649e8d0a844"
_BADWORDS_SHA256 = {
    "ar": "23dbd9c55257bf2a4965f46664dc7eac04b61ff4eb3571e41f7e964c7c3ed96b",
    "cs": "90706df86709bade1ca39cf9f9d9836409c674fc3bffcb7525bdd20e59b596bc",
    "da": "ddb46ab67e955d1a5f49c89c18497d23c3d8d6d2522f3740a5d7583f90e434c6",
    "de": "ebe44d7772897390e27f7bec412fdcdf0efc294e4db157f8487c4cd8d4fabd55",
    "en": "a506cbcf8a31935b122299f3bb1d32be6db3d0ad498765d28f365dfc4f4d7b8c",
    "eo": "0374d76be1dbc77c61362d6e51305e1e4c546b7f734eb96e0bdbad7fd2a6ace1",
    "es": "073334261cc2e7c08339faefd2f19f9623e8240a0ac18fe2a26c368b897496ce",
    "fa": "85828629b90fd18a3f6f6efb1934c1294103c33d1c28c0653970adfa179169e9",
    "fi": "3e0b35b2533ba22fb526d882bea250ddfad907ee7548aacbe15db17e21034c62",
    "fil": "fb4b43691b1249fee2eca1e66795daedda8e7366d42abcde6553dce611d23d99",
    "fr": "79763a705e08256b393952852bb63aa903f4485e0ef02f70735f5e50efff213c",
    "fr-CA-u-sd-caqc": "4a0ef5d55921c91df26676b214d3ac648deef11cfbc4587159ee9aab4085b09a",
    "hi": "83636e9b7a5260451e09e9e471f81dad9f082d107a82da77047e57d7856ed0cc",
    "hu": "a6943a971496f16e9a9d59ba6fde146d229699600e9636828d8d34ed9160aff5",
    "it": "d413f41018d58aa79d2c4809d0161f29ffd5652c4dc34c5b7b9ade680ee5599a",
    "ja": "f83c4d0843ea6474c17bf171edd72b616b3af403f8613f2cde1bd841faf6f316",
    "kab": "df7be2ce4a0c80ee717cd7134c90645a8a0f0c0c458a61f927934dbd50eac96d",
    "ko": "0dd4b74ffea6e3bb35d658e6d8262b53ee99281eaa48ecb515a2065bfcc7f69d",
    "nl": "8298ba0cdc3ca9a1989c349b8ad0148e4504c2804f2f2fb6c2f52bfb39c5ae71",
    "no": "e55b46bcec1ea288256924523a997d8d99e124c9aa3e61cbbbfaeca294f745b4",
    "pl": "98cf824e3692255645c5a72d5913f26fd4f20930f725683b4c96bb312ca5d9b3",
    "pt": "05ef524b0bb7f83f87257f6ec4dabf70e16b8bb6930f769fb93deaf608ff196d",
    "ru": "2598adabd2b761733d72a68d8ee4fcc1d2c38f2f85b6ff489985c6ce1d93bf75",
    "sv": "77f3f99a89bd3b2bde6a654741bacfcb3acca3f5c804af4d8d3fea11e09ea562",
    "th": "835036d87a3a67738e388cd6de92e3c995c186a1427b4e830fe7b85483c257e7",
    "tlh": "3c19004e338c0d91d19ba419185279145848b75b2327aa0e3a752c604ad9f59d",
    "tr": "6c7ae03baee0f3fdc22eed169030567669005dad9a1923e0b571c4bd4f2f2ab5",
    "zh": "a1d9aa037c8b039ef3b40148b3364ce2ca62ce4a955b7082a16ad99f6cbd1bc0",
}
# datatrove's _BADWORDS_ALLOWLIST: removed from every language's list.
_BADWORDS_ALLOWLIST = {"ja": {"sm", "グロ", "女の子"}, "zh": {"性"}}
_NO_BOUNDARY = frozenset(("ja", "th", "zh"))  # matched without word separation
_BADWORDS_REASON = "document_removed_with_badwords"


def _cache_dir() -> Any:
    import os
    from pathlib import Path

    if d := os.environ.get("PYROID_CACHE_DIR"):
        return Path(d)
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "pyroid"


def _badwords_url(language: str) -> str:
    commit = _EN_BADWORDS_COMMIT if language == "en" else _BADWORDS_COMMIT
    return f"{_LDNOOBW}{commit}/{language}"


def _badwords_file(language: str, cache_dir: Any, download: bool) -> Any:
    import hashlib
    import os
    import tempfile
    import urllib.request
    from pathlib import Path

    want = _BADWORDS_SHA256[language]
    target = Path(cache_dir if cache_dir is not None else _cache_dir()) / "c4_badwords" / language
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() == want:
            return target
        if not download:
            raise OSError(f"{target}: sha256 differs from the list datatrove pins")
    elif not download:
        raise FileNotFoundError(f"no {language!r} bad-word list in {target.parent}")
    url = _badwords_url(language)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{language}.")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url) as resp:
            data = resp.read()
            out.write(data)
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            raise OSError(f"{url}: sha256 {got}, expected {want}")
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return target


def load_c4_badwords(
    language: str, *, cache_dir: Any = None, download: bool = True
) -> frozenset[str]:
    """Datatrove's bad-word set for ``language``.

    The lists are the LDNOOBW "List of Dirty, Naughty, Obscene, and Otherwise
    Bad Words" (https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words,
    CC BY 4.0), at the two commits datatrove pins. pyroid does not ship them:
    like datatrove, it downloads a list the first time it is needed, into
    ``cache_dir`` (default ``$PYROID_CACHE_DIR``, else ``$XDG_CACHE_HOME/pyroid``,
    else ``~/.cache/pyroid``) under ``c4_badwords/``, and checks its sha256.
    The set is built as datatrove builds it: each line ``strip()``-ed, minus
    every language's allowlist (``sm``, ``グロ``, ``女の子``, ``性``).

    Args:
        language: One of :data:`C4_BADWORDS_LANGUAGES`.
        cache_dir: Where the lists are kept.
        download: Fetch a missing list (else ``FileNotFoundError``).

    Raises:
        ValueError: No list for ``language`` (datatrove's message).
        OSError: The download failed, or the file's sha256 is not the pinned one.
    """
    if language not in _BADWORDS_SHA256:
        raise ValueError(
            f'There is not badwords list available for "{language}". '
            f"Set fail_on_missing_language=False to continue anyway."
        )
    path = _badwords_file(language, cache_dir, download)
    # datatrove: `badwords.update(line.strip() for line in f)` (text mode), minus every allowlist.
    with open(path, encoding="utf-8") as f:
        words = {line.strip() for line in f}
    for allow in _BADWORDS_ALLOWLIST.values():
        words -= allow
    return frozenset(words)


class C4BadWords(NamedTuple):
    """``C4BadWordsFilter``'s parameters and defaults (``exclusion_writer`` aside).

    Attributes:
        keep_fraction: The fraction of documents with bad words to keep anyway,
            decided per document by ``numpy.random.default_rng(seed).uniform()``.
        fail_on_missing_language: Raise for a language without a list (else keep
            its documents).
        seed: ``default_rng``'s seed: a non-negative int, or ``None`` for fresh OS
            entropy (as numpy).
        default_language: The language of documents given none.
    """

    keep_fraction: float = 0.0
    fail_on_missing_language: bool = True
    seed: int | None = None
    default_language: str = "en"


class BadWordsResult(NamedTuple):
    """What a :class:`BadWordsFilter` call returns, each in the caller's container kind."""

    keep: Any
    reason: Any


def _seed_words(seed: object) -> list[int]:
    if seed is None:
        import secrets

        seed = secrets.randbits(128)  # what numpy's SeedSequence() draws
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int or None, not {type(seed).__name__}")
    if seed < 0:
        raise ValueError("expected non-negative integer")
    words = []
    while seed:
        words.append(seed & 0xFFFFFFFF)
        seed >>= 32
    return words or [0]


class BadWordsFilter:
    r"""datatrove's ``C4BadWordsFilter``, keeping its random state across calls.

    A document is dropped when ``doc.text.lower()`` contains a word of its
    language's list, flanked by non-word characters (``re``'s ``\W``) or the
    text's ends (``ja``, ``th`` and ``zh`` words match anywhere), unless
    ``keep_fraction > 0`` and the next ``uniform()`` draw of numpy's
    ``default_rng(seed)`` is below it. The draws are numpy's PCG64 stream, bit
    for bit, one per flagged document in document order, continuing across
    calls (:attr:`rng_state`). The search runs in Rust, in parallel.

    Attributes:
        config: The :class:`C4BadWords` settings.
        rng_state: ``(state, inc)`` of the PCG64 generator (as numpy's
            ``bit_generator.state["state"]``); assignable.
    """

    def __init__(
        self,
        config: C4BadWords | None = None,
        /,
        *,
        words: Any = None,
        cache_dir: Any = None,
        download: bool = True,
        **kwargs: Any,
    ) -> None:
        """A filter whose generator is ``default_rng(seed)``.

        Args:
            config: A :class:`C4BadWords`; keyword arguments override its fields.
            words: Word lists to use instead of datatrove's, by language (any
                language code; matched literally, not lowercased). Languages
                not given are loaded with :func:`load_c4_badwords` when first needed.
            cache_dir: As for :func:`load_c4_badwords`.
            download: As for :func:`load_c4_badwords`.
            **kwargs: ``C4BadWordsFilter``'s parameters.

        Raises:
            TypeError: ``seed`` is not an int or ``None``.
            ValueError: ``seed`` is negative.
        """
        cfg = _config(C4BadWords, config, kwargs)
        if not isinstance(cfg.default_language, str):
            raise TypeError("default_language must be a str")
        self.config: C4BadWords = cfg
        self._keep_fraction = _num("keep_fraction", cfg.keep_fraction)
        self._words: dict[str, Any] = {} if words is None else dict(words)
        self._cache_dir = cache_dir
        self._download = download
        self._ids_by_lang: dict[str, int] = {}
        self._native: list[Any] = []
        self.rng_state: tuple[int, int] = _native._pcg64_seed(_seed_words(cfg.seed))

    def _list(self, language: str) -> int:
        idx = self._ids_by_lang.get(language)
        if idx is not None:
            return idx
        if language in self._words:
            words = [w for w in self._words[language] if isinstance(w, str)]
        elif language in _BADWORDS_SHA256:
            words = list(
                load_c4_badwords(language, cache_dir=self._cache_dir, download=self._download)
            )
        elif self.config.fail_on_missing_language:
            raise ValueError(
                f'There is not badwords list available for "{language}". '
                f"Set fail_on_missing_language=False to continue anyway."
            )
        else:
            self._ids_by_lang[language] = -1
            return -1
        self._native.append(_native._BadWords(words, language not in _NO_BOUNDARY))
        self._ids_by_lang[language] = idx = len(self._native) - 1
        return idx

    def _run(self, texts: Any, languages: Any, what: str, threads: Any) -> Any:
        if languages is None or isinstance(languages, str):
            lang = self._list(self.config.default_language if languages is None else languages)
            ids = None
        else:
            default = self.config.default_language
            lang = -1
            ids = [self._list(default if x is None else x) for x in languages]
        out, state = _native._badwords_run(
            texts,
            self._native,
            lang,
            ids,
            self._keep_fraction,
            self.rng_state,
            what,
            _unicode_version(),
            threads=threads,
        )
        self.rng_state = state
        return out

    def __call__(self, texts: Any, /, *, languages: Any = None, threads: Any = None) -> Any:
        """Filter ``texts``: keep mask and drop reasons from one pass.

        Args:
            texts: A str, a list or tuple of str, or an Arrow string array/column.
            languages: Each document's language (datatrove's ``doc.metadata["language"]``):
                one str for all, or one entry per document (``None``: the default).
                Default: ``default_language``.
            threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at most
                ``n`` tasks at once; ``1`` runs on the calling thread.

        Returns:
            ``BadWordsResult(keep, reason)`` in the caller's kind (Arrow nulls stay null).

        Raises:
            ValueError: A language has no list and ``fail_on_missing_language``
                (datatrove's message), or ``languages`` has the wrong length.
            OSError: A list could not be downloaded.
            pyroid.InputError: ``texts`` is not one of the accepted kinds.
            UnicodeEncodeError: A str contains a lone surrogate.
            UnicodeDecodeError: An Arrow row is not valid UTF-8.
        """
        keep, reason = self._run(texts, languages, "all", threads)
        return BadWordsResult(keep, reason)

    def keep(self, texts: Any, /, *, languages: Any = None, threads: Any = None) -> Any:
        """The keep mask only (draws as :meth:`__call__`)."""
        return self._run(texts, languages, "keep", threads)

    def reasons(self, texts: Any, /, *, languages: Any = None, threads: Any = None) -> Any:
        """The drop reasons only (draws as :meth:`__call__`)."""
        return self._run(texts, languages, "reason", threads)

    def found(self, texts: Any, /, *, languages: Any = None, threads: Any = None) -> Any:
        """Whether each document contains a bad word (no draws; missing lists give False)."""
        return self._run(texts, languages, "found", threads)

    def __repr__(self) -> str:
        return f"BadWordsFilter({self.config!r})"


def c4_badwords(
    texts: Any,
    config: Any = None,
    /,
    *,
    languages: Any = None,
    words: Any = None,
    threads: Any = None,
    **kwargs: Any,
) -> Any:
    """``C4BadWordsFilter``: True for each document datatrove keeps.

    A fresh :class:`BadWordsFilter` per call, so the draws start from
    ``default_rng(seed)``: one call over a whole corpus gives datatrove's
    decisions. For batches, use one :class:`BadWordsFilter`.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        config: A :class:`C4BadWords`; keyword arguments override its fields.
        languages: As for :meth:`BadWordsFilter.__call__`.
        words: As for :class:`BadWordsFilter`.
        threads: As for :meth:`BadWordsFilter.__call__`.
        **kwargs: ``C4BadWordsFilter``'s parameters.

    Returns:
        A keep mask in the caller's kind.
    """
    return BadWordsFilter(config, words=words, **kwargs).keep(
        texts, languages=languages, threads=threads
    )


def c4_badwords_reasons(
    texts: Any,
    config: Any = None,
    /,
    *,
    languages: Any = None,
    words: Any = None,
    threads: Any = None,
    **kwargs: Any,
) -> Any:
    """``C4BadWordsFilter``: datatrove's drop reason per document, ``None`` when kept.

    As :func:`c4_badwords`, returning ``"document_removed_with_badwords"`` or ``None``.
    """
    return BadWordsFilter(config, words=words, **kwargs).reasons(
        texts, languages=languages, threads=threads
    )


# ---- end C4BadWordsFilter ----

# ---- URLFilter ----
import os as _os  # noqa: E402
import re as _re  # noqa: E402
import sys as _sys  # noqa: E402

from pyroid._pyroid import FallbackWarning as _FallbackWarning  # noqa: E402
from pyroid._pyroid import PyroidError as _PyroidError  # noqa: E402

__all__ += [
    "TLDEXTRACT_VERSION",
    "DatatroveURLLists",
    "URLFilter",
    "URLFilterConfig",
    "datatrove_url_lists",
    "url_filter",
    "url_filter_reasons",
]

#: The tldextract release whose ``extract_str`` the URL filter reproduces.
TLDEXTRACT_VERSION = "5.4.0"

_URL_ASSET = "url_filterblacklistsv0_3_0.tar.gz"
_URL_NORMALIZER = _re.compile(r"[^a-zA-Z0-9]+")
_NEVER = 2**64 - 1
_NO_AUTOMATON = (
    "Not an Aho-Corasick automaton yet: call add_word to add some keys and call "
    "make_automaton to convert the trie to an automaton."
)
_url_fallbacks: set[str] = set()


def _url_fallback(reason: str) -> None:
    if reason not in _url_fallbacks:
        _url_fallbacks.add(reason)
        warnings.warn(
            f"pyroid.filters.URLFilter: {reason} is not supported by pyroid; running "
            "tldextract's own code for them (slow path). This warning is shown once.",
            _FallbackWarning,
            stacklevel=_caller_stacklevel(),
        )


def _url_normalize(text: Any, replace: str = "") -> Any:
    """Normalize as datatrove's ``url_filter.normalize``."""
    return _URL_NORMALIZER.sub(replace, text).lower()


def _parse_list(line: Iterable[Any], do_normalize: bool = True) -> set[Any]:
    """Parse as datatrove's ``url_filter.parse_list``, verbatim (same results and exceptions)."""
    return {_url_normalize(x) if do_normalize else x.strip() for x in line if x[0] != "#"}


def _text_encoding() -> str:
    """What ``open(path)`` decodes with (without ``-X warn_default_encoding``'s warning)."""
    return "utf-8" if _sys.flags.utf8_mode else "locale"


def _get_list(path: str, extra: set[Any], do_normalize: bool = True) -> set[Any]:
    """Read a list file as datatrove's ``url_filter.get_list``."""
    with open(path, encoding=_text_encoding()) as f:
        return _parse_list(f, do_normalize).union(extra)


def _read_text(path: str) -> str:
    """The text ``open(path)`` iterates over (newlines translated)."""
    with open(path, encoding=_text_encoding()) as f:
        return f.read()


def _encodable(s: str) -> bool:
    """No lone surrogate (pyroid takes UTF-8 strings only)."""
    if s.isascii():
        return True
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


class URLFilterConfig(NamedTuple):
    """datatrove 0.10.1 ``URLFilter``'s parameters and defaults.

    The lists are read as datatrove's ``parse_list`` reads them: entries starting
    with ``#`` are skipped (the test is ``x[0] != "#"``, so an empty string raises
    ``IndexError``, as in datatrove); domains and URLs are ``strip()``-ped; words,
    subwords and soft words are normalized (every character but ``[a-zA-Z0-9]``
    removed, then lowercased), so ``"--"`` becomes ``""``. A falsy list adds nothing.
    """

    soft_word_threshold: Any = 2
    extra_domains: Iterable[str] | None = None
    extra_urls: Iterable[str] | None = None
    banned_words: Iterable[str] | None = None
    banned_subwords: Iterable[str] | None = None
    soft_banned_words: Iterable[str] | None = None
    use_integrated_lists: bool = True


class DatatroveURLLists(NamedTuple):
    """The files of datatrove's integrated URL block lists (see :func:`datatrove_url_lists`)."""

    domains: str
    urls: str
    banned_words: str
    banned_subwords: str
    soft_banned_words: str


def datatrove_url_lists() -> DatatroveURLLists:
    """Extract datatrove's integrated URL block lists as datatrove does; return their paths.

    datatrove 0.10.1 ships them in its package (``datatrove/assets``):
    ``url_filterblacklistsv0_3_0.tar.gz`` (``domains``: 4,558,940 lines; ``urls``:
    19,587), ``banned_words.txt``, ``banned_subwords.txt`` and
    ``soft_banned_words.txt``. The archive is extracted once into datatrove's own cache
    (``huggingface_hub.cached_assets_path("datatrove", "filters", "url_filter")``),
    under datatrove's lock and ``.completed`` marker, so datatrove and pyroid share
    one copy. pyroid does not ship these lists: they are not under pyroid's licence.

    Raises:
        pyroid.PyroidError: datatrove (or huggingface_hub) is not installed.
        ImportError: fasteners is not installed (datatrove's lock needs it).
    """
    try:
        from datatrove.io import safely_create_file
        from datatrove.utils._import_utils import ASSETS_PATH
        from huggingface_hub import cached_assets_path
    except ImportError as e:
        raise _PyroidError(
            "use_integrated_lists=True reads the block lists shipped with datatrove: install "
            "datatrove==0.10.1, or pass use_integrated_lists=False and your own lists"
        ) from e
    import tarfile

    download_dir = cached_assets_path(
        library_name="datatrove", namespace="filters", subfolder="url_filter"
    )
    file_to_lock = _os.path.join(download_dir, _URL_ASSET)

    def do_extract() -> None:
        with tarfile.open(_os.path.join(ASSETS_PATH, _URL_ASSET), "r:gz") as tar:
            if hasattr(tarfile, "data_filter"):
                # The same files as datatrove's bare extractall(), without the
                # DeprecationWarning of Python 3.12-3.13.
                tar.extractall(download_dir, filter="data")
            else:  # pragma: no cover - Python < 3.11.4
                tar.extractall(download_dir)

    safely_create_file(file_to_lock, do_extract)
    return DatatroveURLLists(
        domains=_os.path.join(download_dir, "domains"),
        urls=_os.path.join(download_dir, "urls"),
        banned_words=_os.path.join(ASSETS_PATH, "banned_words.txt"),
        banned_subwords=_os.path.join(ASSETS_PATH, "banned_subwords.txt"),
        soft_banned_words=_os.path.join(ASSETS_PATH, "soft_banned_words.txt"),
    )


def _min_soft_count(threshold: Any) -> int:
    """The smallest count ``n`` with ``n >= soft_word_threshold`` true (``_NEVER``: none).

    datatrove compares the (int) number of soft words with the threshold; this asks
    Python's own ``>=`` (so ints, floats, fractions, NaN, infinities and numpy scalars
    behave as in datatrove), by bisection over ``[0, 2**63]``.
    """
    try:
        if bool(0 >= threshold):  # noqa: SIM300 - datatrove evaluates `int >= threshold`
            return 0
        lo, hi = 0, 2**63
        if not bool(hi >= threshold):
            return _NEVER
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if bool(mid >= threshold):
                hi = mid
            else:
                lo = mid
    except TypeError as e:
        raise TypeError(
            f"soft_word_threshold must compare with an int, not {type(threshold).__name__}"
        ) from e
    return hi


def _tld_extractor(tld: Any) -> Any:
    try:
        import tldextract
    except ImportError as e:
        if tld is None:
            raise _PyroidError(
                "pyroid.filters.URLFilter needs tldextract (its Public Suffix List and "
                f"cache): pip install tldextract=={TLDEXTRACT_VERSION}"
            ) from e
    else:
        # The URL filter runs on tldextract's suffix list: checked when it loads tldextract.
        _versions.check(
            "pyroid.filters.URLFilter", "tldextract", f"tldextract=={TLDEXTRACT_VERSION}"
        )
        if tld is None:
            tld = tldextract.TLDExtract()
    return tld


def _decode_label(lowered: str) -> str:
    """Decode as tldextract 5.4.0's ``_decode_punycode`` does after its ``label.lower()``."""
    import idna

    if lowered.startswith("xn--"):
        try:
            return str(idna.decode(lowered))
        except (UnicodeError, IndexError):
            pass
    return lowered


def _suffix_trie(tld: Any) -> tuple[Any, ...]:
    """The trie ``tld.suffix_index`` walks, and the lowered labels that decode to its keys."""
    ex = tld._get_tld_extractor()
    root = (
        ex.tlds_incl_private_trie if ex.include_psl_private_domains else ex.tlds_excl_private_trie
    )
    strings: list[str] = []
    ids: dict[str, int] = {}

    def intern(s: str) -> int:
        i = ids.get(s)
        if i is None:
            i = ids[s] = len(strings)
            strings.append(s)
        return i

    nodes = [(bool(root.end), bool(root.is_private))]
    edges: list[tuple[int, int, int]] = []
    stack = [(root, 0)]
    while stack:
        node, nid = stack.pop()
        for key, child in node.matches.items():
            if not isinstance(key, str) or not _encodable(key):
                continue  # no label of a UTF-8 URL decodes to it
            cid = len(nodes)
            nodes.append((bool(child.end), bool(child.is_private)))
            edges.append((nid, intern(str(key)), cid))
            stack.append((child, cid))
    for s in [s[1:] for s in strings if s.startswith("!")]:
        intern(s)
    self_ids = [i for i, s in enumerate(strings) if _decode_label(s) == s]
    aliases = []
    for i, s in enumerate(strings):
        try:
            a = "xn--" + s.encode("punycode").decode("ascii")
        except UnicodeError:
            continue
        if a != s and _decode_label(a) == s:
            aliases.append((a, i))
    # idna >= 3.?: an A-label decodes only if it is the canonical encoding of its U-label.
    canonical = _decode_label("xn--bbk") != "xn--bbk" and _decode_label("xn---bbk") == "xn---bbk"
    return strings, self_ids, aliases, nodes, edges, not canonical


def _resolver(tld: Any) -> Any:
    def resolve(url: str) -> tuple[str, str]:
        r = tld(url)
        top = getattr(type(r), "top_domain_under_public_suffix", None)  # tldextract >= 5.3
        if top is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                return r.registered_domain, r.fqdn
        return r.top_domain_under_public_suffix, r.fqdn

    return resolve


class URLFilter:
    """datatrove 0.10.1's ``URLFilter``, built once: the same decisions and drop reasons.

    Build it once and reuse it: with the integrated lists it holds 4.56 million
    blocked domains (about 200 MB, in one Rust string arena with a hash table),
    and building it takes a second or two. Each URL is checked as datatrove does,
    in its order, and the first hit is the reason:

    1. ``"domain"``: tldextract's registered domain
       (``top_domain_under_public_suffix``) is a blocked domain;
    2. ``"subdomain"``: its FQDN is a blocked domain;
    3. ``"url"``: the URL itself is blocked;
    4. ``"hard_blacklisted"``: a word of ``re.split(r"[^a-zA-Z0-9]+", url)`` (case
       sensitive; ``""`` when the URL starts or ends with another character) is
       a banned word;
    5. ``"soft_blacklisted"``: at least ``soft_word_threshold`` distinct words are
       soft-banned;
    6. ``"blacklisted_subword"``: a banned subword is a substring of the URL with
       every character but ``[a-zA-Z0-9]`` removed, lowercased.

    Domains come from tldextract 5.4.0's ``extract_str``, ported to Rust:
    ``lenient_netloc`` (scheme, userinfo, ``[IPv6]``, port, ``strip()``, trailing
    dots), U+3002, U+FF0E and U+FF61 as dots, the IPv6 literal test (``ipaddress.IPv6Address``), and
    the Public Suffix List walk (wildcards such as ``*.ck``, exceptions such as
    ``!www.ck``), labels lowered with ``str.lower()`` of the running interpreter's
    Unicode version and ``xn--`` labels decoded as ``idna`` does. The suffix lists
    are the ones ``tld_extractor`` uses (its cached copy of the Public Suffix List,
    or tldextract's bundled snapshot), read from it when the filter is built;
    with its ``include_psl_private_domains`` setting.

    Args:
        config: A :class:`URLFilterConfig`; keyword arguments override its fields.
        tld_extractor: A ``tldextract.TLDExtract`` (default: a new
            ``TLDExtract()``, as datatrove builds).
        **kwargs: ``URLFilter``'s parameters (see :class:`URLFilterConfig`).

    Raises:
        TypeError: ``config`` is not a :class:`URLFilterConfig`, or
            ``soft_word_threshold`` does not compare with an int (datatrove raises
            this ``TypeError`` at the first URL that reaches the soft-word check).
        IndexError: A list has an empty string (as datatrove).
        pyroid.PyroidError: tldextract is not installed, or
            ``use_integrated_lists=True`` and datatrove is not installed.

    The methods take URLs as a str, a list or tuple of str, or an Arrow string
    array/column, with ``threads=`` as everywhere in pyroid, and raise
    ``ValueError`` for an empty URL (datatrove's ``assert url``),
    ``AttributeError`` as datatrove when the banned subwords are only ``""``
    (pyahocorasick's automaton then has no words) and a URL reaches that check,
    ``pyroid.InputError`` for other input kinds, ``UnicodeEncodeError`` for a lone
    surrogate and ``UnicodeDecodeError`` for invalid UTF-8 in Arrow input.
    """

    config: URLFilterConfig | None

    def __init__(self, config: Any = None, /, *, tld_extractor: Any = None, **kwargs: Any):
        cfg: URLFilterConfig = _config(URLFilterConfig, config, kwargs)
        self._build(
            cfg.soft_word_threshold,
            _parse_list(cfg.extra_domains, do_normalize=False) if cfg.extra_domains else set(),
            _parse_list(cfg.extra_urls, do_normalize=False) if cfg.extra_urls else set(),
            _parse_list(cfg.banned_words) if cfg.banned_words else set(),
            _parse_list(cfg.banned_subwords) if cfg.banned_subwords else set(),
            _parse_list(cfg.soft_banned_words) if cfg.soft_banned_words else set(),
            cfg.use_integrated_lists,
            tld_extractor,
        )
        self.config = cfg

    @classmethod
    def _from_parsed(
        cls,
        soft_word_threshold: Any,
        domains: set[Any],
        urls: set[Any],
        banned_words: set[Any],
        banned_subwords: set[Any],
        soft_banned_words: set[Any],
        use_integrated_lists: Any,
        tld_extractor: Any,
    ) -> URLFilter:
        """From sets already parsed as datatrove's ``__init__`` does (the drop-in step)."""
        self = cls.__new__(cls)
        self._build(
            soft_word_threshold,
            domains,
            urls,
            banned_words,
            banned_subwords,
            soft_banned_words,
            use_integrated_lists,
            tld_extractor,
        )
        self.config = None
        return self

    def _build(
        self,
        threshold: Any,
        domains: set[Any],
        urls: set[Any],
        banned_words: set[Any],
        banned_subwords: set[Any],
        soft_banned_words: set[Any],
        use_integrated_lists: Any,
        tld_extractor: Any,
    ) -> None:
        tld = _tld_extractor(tld_extractor)
        min_soft = _min_soft_count(threshold)
        domain_texts: list[str] = []
        url_texts: list[str] = []
        if use_integrated_lists:  # datatrove's download_data()
            paths = datatrove_url_lists()
            domain_texts.append(_read_text(paths.domains))
            url_texts.append(_read_text(paths.urls))
            banned_words = _get_list(paths.banned_words, banned_words)
            banned_subwords = _get_list(paths.banned_subwords, banned_subwords)
            soft_banned_words = _get_list(paths.soft_banned_words, soft_banned_words)
        # Only str entries can equal a str; those with lone surrogates stay here, for
        # the Python path (URLs pyroid can't take).
        self._odd_domains = {s for s in domains if isinstance(s, str) and not _encodable(s)}
        self._odd_urls = {s for s in urls if isinstance(s, str) and not _encodable(s)}
        self._threshold = threshold
        self._banned_words = banned_words
        self._banned_subwords = banned_subwords
        self._soft_banned_words = soft_banned_words
        self._tld = tld
        trie = _suffix_trie(tld)
        if trie[-1]:
            _url_fallback("xn-- labels with an idna lacking the canonical A-label check")
        self._native = _native._URLFilter(
            trie,
            ([str(s) for s in domains if isinstance(s, str) and _encodable(s)], domain_texts),
            ([str(s) for s in urls if isinstance(s, str) and _encodable(s)], url_texts),
            (
                [str(w) for w in banned_words],
                [str(w) for w in soft_banned_words],
                [str(w) for w in banned_subwords],
                min_soft,
            ),
            _unicode_version(),
            _resolver(tld),
        )

    def keep(self, urls: Any, /, *, threads: Any = None) -> Any:
        """True for each URL datatrove keeps.

        Returns:
            bool, ``list[bool]``, ``tuple[bool, ...]``, or a :class:`pyroid.Column`
            of bools (nulls stay null), matching ``urls``.
        """
        return self._native.keep(urls, threads=threads)

    def reasons(self, urls: Any, /, *, threads: Any = None) -> Any:
        """The drop reason (datatrove's) per URL, ``None`` when kept.

        Reasons: ``domain``, ``subdomain``, ``url``, ``hard_blacklisted``,
        ``soft_blacklisted``, ``blacklisted_subword``. Arrow input gives a
        ``string`` :class:`pyroid.Column` (null when kept or null).
        """
        return self._native.reasons(urls, threads=threads)

    def domains(self, urls: Any, /, *, threads: Any = None) -> tuple[Any, Any]:
        """The ``(top_domain_under_public_suffix, fqdn)`` tldextract gives each URL.

        What datatrove looks up in the blocked domains (``registered_domain`` is
        ``top_domain_under_public_suffix``). Both are ``""`` for IP addresses,
        hosts without a public suffix, and bare suffixes.

        Returns:
            A pair, each in the kind of ``urls`` (Arrow: ``large_string`` columns
            for ``large_string`` input, else ``string``; nulls stay null).
        """
        rd, fqdn = self._native.domains(urls, threads=threads)
        return rd, fqdn

    @property
    def tld_extractor(self) -> Any:
        """The ``tldextract.TLDExtract`` whose suffix lists are used."""
        return self._tld

    @property
    def num_domains(self) -> int:
        """Number of blocked domains."""
        return int(self._native.num_domains) + len(self._odd_domains)

    @property
    def num_urls(self) -> int:
        """Number of blocked URLs."""
        return int(self._native.num_urls) + len(self._odd_urls)

    def __repr__(self) -> str:
        return f"<pyroid.filters.URLFilter: {self.num_domains} domains, {self.num_urls} urls>"

    def _reasons_any(self, urls: list[Any], threads: Any) -> list[str | None]:
        """Reasons for a datatrove batch: str URLs natively, others in Python.

        Empty URLs are filtered (datatrove asserts them away, unless ``python -O``).
        """
        if all(type(u) is str for u in urls):
            try:
                out: list[str | None] = self._native.reasons(
                    urls, threads=threads, allow_empty=True
                )
            except UnicodeEncodeError:
                pass
            else:
                return out
        res: list[str | None] = []
        for u in urls:
            if type(u) is str and _encodable(u):
                res.append(self._native.reasons(u, allow_empty=True))
                continue
            if isinstance(u, str):
                _url_fallback("URLs with lone surrogates")
            res.append(self._python_reason(u))
        return res

    def _python_reason(self, url: Any) -> str | None:
        """Run datatrove's ``filter()`` in Python, for a URL the native path can't take."""
        top, fqdn = _resolver(self._tld)(url)  # raises as datatrove's tldextractor(url)
        if self._has(top, self._odd_domains, self._native.contains_domain):
            return "domain"
        if self._has(fqdn, self._odd_domains, self._native.contains_domain):
            return "subdomain"
        if self._has(url, self._odd_urls, self._native.contains_url):
            return "url"
        url_words = set(_URL_NORMALIZER.split(url))
        if any(word in url_words for word in self._banned_words):
            return "hard_blacklisted"
        nb_soft_words = sum([word in url_words for word in self._soft_banned_words])
        if nb_soft_words >= self._threshold:
            return "soft_blacklisted"
        normalized_space = _url_normalize(url)
        if self._banned_subwords:
            words = [w for w in self._banned_subwords if w]
            if not words:
                raise AttributeError(_NO_AUTOMATON)
            if any(w in normalized_space for w in words):
                return "blacklisted_subword"
        return None

    @staticmethod
    def _has(s: Any, odd: set[Any], native: Any) -> bool:
        if isinstance(s, str) and _encodable(s):
            return bool(native(s))
        return s in odd


_url_filters: dict[Any, URLFilter] = {}


def _url_key(cfg: URLFilterConfig) -> Any:
    """A cache key for ``cfg``, or ``None`` (one-shot iterables, unhashable values)."""

    def freeze(v: Any) -> Any:
        if v is None or isinstance(v, (str, bool, int, float)):
            return (type(v), v)
        if isinstance(v, (list, tuple, set, frozenset)):
            return (frozenset, frozenset(v))  # parse_list only sees the distinct items
        raise TypeError

    try:
        key = tuple(freeze(v) for v in cfg)
        hash(key)
    except TypeError:
        return None
    return key


def _url_filter(config: Any, kwargs: dict[str, Any]) -> URLFilter:
    if isinstance(config, URLFilter):
        if kwargs:
            raise TypeError("pass a URLFilter or URLFilter's parameters, not both")
        return config
    cfg: URLFilterConfig = _config(URLFilterConfig, config, kwargs)
    key = _url_key(cfg)
    f = None if key is None else _url_filters.get(key)
    if f is None:
        f = URLFilter(cfg)
        if key is not None:
            if len(_url_filters) >= 2:
                _url_filters.clear()
            _url_filters[key] = f
    return f


def url_filter(urls: Any, config: Any = None, /, *, threads: Any = None, **kwargs: Any) -> Any:
    """``URLFilter``: True for each URL datatrove keeps.

    ``config`` is a :class:`URLFilterConfig` (keyword arguments override its
    fields) or a built :class:`URLFilter`. The last two filters built from configs
    are cached (configs whose lists are lists, tuples, sets or strings), so repeated
    calls do not reload the block lists.

    Returns:
        As :meth:`URLFilter.keep`.
    """
    return _url_filter(config, kwargs).keep(urls, threads=threads)


def url_filter_reasons(
    urls: Any, config: Any = None, /, *, threads: Any = None, **kwargs: Any
) -> Any:
    """``URLFilter``: datatrove's drop reason per URL, ``None`` when kept.

    Arguments as for :func:`url_filter`; results as :meth:`URLFilter.reasons`.
    """
    return _url_filter(config, kwargs).reasons(urls, threads=threads)


# ---- end URLFilter ----

# ---- SymbolLinesFormatter ----
__all__ += ["SymbolLines", "symbol_lines"]


class SymbolLines(NamedTuple):
    r"""``SymbolLinesFormatter``'s parameters and defaults.

    Attributes:
        symbols_to_remove: The symbols a removed line may consist of (with
            ``" "``); ``None`` is datatrove's ``PUNCTUATION_SET``.
        replace_char: What a run of removed lines becomes (``""``: nothing;
            FineWeb-2 uses ``"\n"``).
    """

    symbols_to_remove: Iterable[str] | None = None
    replace_char: str = ""


_symbol_cache: dict[Any, Any] = {}


def _symbol_lines_native(cfg: SymbolLines) -> Any:
    if cfg.symbols_to_remove is None:
        chars = None
    else:
        # datatrove tests `c in set(symbols_to_remove)` per character: only 1-char strings match.
        chars = sorted({s for s in cfg.symbols_to_remove if isinstance(s, str) and len(s) == 1})
    if not isinstance(cfg.replace_char, str):
        raise TypeError(f"replace_char must be a str, not {type(cfg.replace_char).__name__}")
    key = (None if chars is None else tuple(chars), cfg.replace_char)
    native = _symbol_cache.get(key)
    if native is None:
        native = _native._SymbolLines(chars, cfg.replace_char)
        if len(_symbol_cache) >= 64:
            _symbol_cache.clear()
        _symbol_cache[key] = native
    return native


def symbol_lines(texts: Any, config: Any = None, /, *, threads: Any = None, **kwargs: Any) -> Any:
    r"""``SymbolLinesFormatter``: each document without its symbol-only lines.

    As datatrove: over ``text.splitlines()``, a non-blank line made only of
    ``symbols_to_remove`` and spaces is dropped, a run of them becomes one
    ``replace_char`` (if not empty), and the kept lines are joined with
    ``"\n"`` (so ``"\r\n"`` and a trailing newline come out as datatrove's do).

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        config: A :class:`SymbolLines`; keyword arguments override its fields.
        threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at most
            ``n`` tasks at once; ``1`` runs on the calling thread.
        **kwargs: ``SymbolLinesFormatter``'s parameters.

    Returns:
        The texts in the caller's kind; a document that comes out equal comes
        back as the same object (Arrow: a :class:`pyroid.Column`, nulls stay null).

    Raises:
        pyroid.InputError: ``texts`` is not one of the accepted kinds.
        TypeError: ``replace_char`` is not a str.
        UnicodeEncodeError: A str contains a lone surrogate.
        UnicodeDecodeError: An Arrow row is not valid UTF-8.
    """
    cfg = _config(SymbolLines, config, kwargs)
    return _symbol_lines_native(cfg).run(texts, threads=threads)


# ---- end SymbolLinesFormatter ----
