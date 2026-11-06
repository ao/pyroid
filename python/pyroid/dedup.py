# Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.
"""MinHash near-duplicate and exact deduplication (``pyroid.dedup``).

The MinHash pipeline makes the same removals as datatrove 0.10.1's
``MinhashDedupSignature`` -> ``MinhashDedupBuckets`` -> ``MinhashDedupCluster``
-> ``MinhashDedupFilter``, in memory. Types and docs are in ``dedup.pyi``.
The language table and the opt-in Python-tokenizer path are shared with
``pyroid.filters``.
"""

from __future__ import annotations

import operator
import unicodedata
import warnings
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from typing import Any

from pyroid import filters as _filters
from pyroid._pyroid import dedup as _native
from pyroid._pyroid import ftfy as _ftfy_native
from pyroid.filters import Language, languages

__all__ = [
    "DATATROVE_VERSION",
    "ExactDedup",
    "HashConfig",
    "Language",
    "MinHashDedup",
    "MinhashConfig",
    "TextNormConfig",
    "exact_dedup",
    "languages",
    "minhash_dedup",
    "minhash_signatures",
    "simplify_text",
]

#: The datatrove release whose decisions this module reproduces.
DATATROVE_VERSION = "0.10.1"


def _unicode_version() -> str:
    """The interpreter's Unicode version, or the newest one the kernel knows."""
    version = unicodedata.unidata_version
    supported = _ftfy_native.supported_unicode_versions()
    if version in supported:
        return version
    newest = supported[-1]
    warnings.warn(
        f"pyroid.dedup has no tables for Unicode {version} (this Python's unicodedata); "
        f"using Unicode {newest}. simplify_text can differ from datatrove for characters "
        f"added after {newest}.",
        RuntimeWarning,
        stacklevel=3,
    )
    return newest


_UNICODE_VERSION = _unicode_version()


@dataclass(frozen=True)
class HashConfig:
    """datatrove's ``HashConfig``: the shingle hash function and its width."""

    precision: int = 64
    hash_fc: str = "xxhash"


@dataclass(frozen=True)
class TextNormConfig:
    """datatrove's ``TextNormConfig``: what ``simplify_text`` does."""

    lowercase: bool = True
    norm_whitespace: bool = True
    remove_punctuation: bool = True
    norm_unicode_diacritics: bool = True
    norm_numbers: bool = True
    norm_weekdays: bool = False
    norm_monthnames: bool = False


@dataclass(frozen=True)
class MinhashConfig:
    """datatrove's ``MinhashConfig``, with the same fields and defaults."""

    n_grams: int = 5
    num_buckets: int = 14
    hashes_per_bucket: int = 8
    seed: int = 1
    norm_config: TextNormConfig = field(default_factory=TextNormConfig)
    hash_config: HashConfig = field(default_factory=HashConfig)


_HASH_FIELDS = tuple(f.name for f in fields(HashConfig))
_NORM_FIELDS = tuple(f.name for f in fields(TextNormConfig))
_CONFIG_FIELDS = tuple(f.name for f in fields(MinhashConfig))


def _int(name: str, value: object, lo: int, hi: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got bool")
    try:
        v = operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}") from None
    if not lo <= v <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}, got {v}")
    return v


def _from_fields(cls: Any, obj: object, names: tuple[str, ...]) -> Any:
    """``cls`` from a pyroid instance, a dict, or any object with the fields (datatrove's)."""
    if isinstance(obj, cls):
        return obj
    if isinstance(obj, dict):
        unknown = set(obj) - set(names)
        if unknown:
            raise TypeError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
        return cls(**obj)
    try:
        return cls(**{n: getattr(obj, n) for n in names})
    except AttributeError:
        raise TypeError(
            f"expected a {cls.__name__} (pyroid's or datatrove's), got {type(obj).__name__}"
        ) from None


def _norm_config(obj: object) -> TextNormConfig:
    norm: TextNormConfig = (
        TextNormConfig() if obj is None else _from_fields(TextNormConfig, obj, _NORM_FIELDS)
    )
    return TextNormConfig(**{n: bool(getattr(norm, n)) for n in _NORM_FIELDS})


def _hash_config(obj: object) -> HashConfig:
    hc: HashConfig = HashConfig() if obj is None else _from_fields(HashConfig, obj, _HASH_FIELDS)
    if isinstance(hc.precision, bool) or hc.precision not in (32, 64):
        raise ValueError(f"precision must be 32 or 64, got {hc.precision!r}")
    if hc.hash_fc not in ("sha1", "xxhash"):
        raise ValueError(f"hash_fc must be 'sha1' or 'xxhash', got {hc.hash_fc!r}")
    return HashConfig(precision=int(hc.precision), hash_fc=str(hc.hash_fc))


def _minhash_config(config: object, kwargs: dict[str, Any]) -> MinhashConfig:
    """``config`` (pyroid's, datatrove's, a dict or None) with ``kwargs`` applied.

    ``kwargs`` are ``MinhashConfig`` fields, plus ``hash_fc`` / ``precision`` as
    shorthands for ``hash_config``.
    """
    if config is None:
        cfg = MinhashConfig()
    elif isinstance(config, MinhashConfig):
        cfg = config
    elif isinstance(config, dict):
        kwargs = {**config, **kwargs}
        cfg = MinhashConfig()
    else:
        cfg = _from_fields(MinhashConfig, config, _CONFIG_FIELDS)
    if kwargs:
        hash_kw = {k: kwargs.pop(k) for k in _HASH_FIELDS if k in kwargs}
        unknown = set(kwargs) - set(_CONFIG_FIELDS)
        if unknown:
            raise TypeError(f"unknown MinhashConfig fields: {sorted(unknown)}")
        cfg = replace(cfg, **kwargs)
        if hash_kw:
            cfg = replace(cfg, hash_config=replace(_hash_config(cfg.hash_config), **hash_kw))
    return MinhashConfig(
        n_grams=_int("n_grams", cfg.n_grams, 1, 2**31),
        num_buckets=_int("num_buckets", cfg.num_buckets, 1, 2**20),
        hashes_per_bucket=_int("hashes_per_bucket", cfg.hashes_per_bucket, 1, 2**20),
        seed=_int("seed", cfg.seed, 0, 2**32 - 1),
        norm_config=_norm_config(cfg.norm_config),
        hash_config=_hash_config(cfg.hash_config),
    )


@lru_cache(maxsize=64)
def _native_norm(norm: TextNormConfig) -> Any:
    return _native._NativeNorm(
        **{n: getattr(norm, n) for n in _NORM_FIELDS}, unicode_version=_UNICODE_VERSION
    )


@lru_cache(maxsize=64)
def _native_config(cfg: MinhashConfig, language: str | None = None) -> Any:
    return _native._NativeConfig(
        n_grams=cfg.n_grams,
        num_buckets=cfg.num_buckets,
        hashes_per_bucket=cfg.hashes_per_bucket,
        seed=cfg.seed,
        hash_fc=cfg.hash_config.hash_fc,
        precision=cfg.hash_config.precision,
        norm=_native_norm(cfg.norm_config),
        language=language,
    )


def _normalize_arg(normalize: object) -> Any:
    if normalize is None or normalize is False:
        return None
    if normalize is True:
        return _native_norm(TextNormConfig())
    return _native_norm(_norm_config(normalize))


_MATH = "shingling and MinHash still run in Rust"


def _words_source(language: object, python_tokenizer: object, where: str) -> Any:
    """``language`` as a native code (``str``), or the Python tokenizer to use.

    Raises datatrove's ``ValueError`` for an unknown code, ``NotImplementedError`` for one
    pyroid does not tokenize natively without ``python_tokenizer=True``, ``PyroidError``
    if that path's packages are missing.
    """
    lang = _filters._language(language)
    if isinstance(lang, str) and not python_tokenizer:
        _filters._native_language(lang, _MATH)
        return lang
    tok = _filters._load_tokenizer(lang, where)
    _filters._warn_tokenizer(where.removeprefix("pyroid."), tok, _MATH)
    return tok


def _python_tokens(
    texts: Any, cfg: MinhashConfig, tok: Any, threads: int | None, where: str
) -> tuple[list[list[str]], Any, list[Any]]:
    """The ``word_tokenize(simplify_text(text))`` of datatrove, per document (``[]`` for nulls).

    Returns the tokens, the documents' shape and the documents (``None`` for Arrow nulls).
    """
    docs, shape = _filters._native._to_list(texts, threads=threads)
    at = [i for i, d in enumerate(docs) if d is not None]
    present = [docs[i] for i in at]
    simplified = _native._simplify_text(present, _native_norm(cfg.norm_config), threads=threads)
    words = iter(_filters._tokenize(tok, simplified, where, at))
    return [[] if d is None else next(words) for d in docs], shape, docs


def _mask(shape: Any, docs: list[Any], keep: list[bool], threads: int | None) -> Any:
    """``keep`` in the caller's kind (``None`` for Arrow nulls)."""
    mask = [None if d is None else k for d, k in zip(docs, keep, strict=True)]
    blank = [None] * len(mask)
    return _filters._native._pack(shape, "keep", mask, blank, blank, threads=threads)


def minhash_signatures(
    texts: Any,
    config: Any = None,
    *,
    threads: int | None = None,
    language: Any = "eng",
    python_tokenizer: bool = False,
    **kwargs: Any,
) -> Any:
    """MinHash signature of each document, as datatrove computes it (see ``dedup.pyi``)."""
    cfg = _minhash_config(config, kwargs)
    src = _words_source(language, python_tokenizer, "pyroid.dedup.minhash_signatures")
    if isinstance(src, str):
        return _native._minhash_signatures(texts, _native_config(cfg, src), threads=threads)
    tokens, _, _ = _python_tokens(texts, cfg, src, threads, "pyroid.dedup.minhash_signatures")
    sigs = _native._signatures_from_tokens(tokens, _native_config(cfg), threads=threads)
    return sigs[0] if isinstance(texts, str) else sigs


def minhash_dedup(
    texts: Any,
    config: Any = None,
    *,
    threads: int | None = None,
    language: Any = "eng",
    python_tokenizer: bool = False,
    **kwargs: Any,
) -> Any:
    """Keep mask of datatrove's MinHash dedup over ``texts`` (see ``dedup.pyi``)."""
    cfg = _minhash_config(config, kwargs)
    src = _words_source(language, python_tokenizer, "pyroid.dedup.minhash_dedup")
    if isinstance(src, str):
        return _native._MinHashState(_native_config(cfg, src), threads=threads).fit(texts)
    state = _native._MinHashState(_native_config(cfg), threads=threads)
    tokens, shape, docs = _python_tokens(texts, cfg, src, threads, "pyroid.dedup.minhash_dedup")
    return _mask(shape, docs, state._fit_tokens(tokens), threads)


def simplify_text(texts: Any, config: Any = None, *, threads: int | None = None) -> Any:
    """The ``simplify_text`` normalisation of datatrove (see ``dedup.pyi``)."""
    return _native._simplify_text(texts, _native_norm(_norm_config(config)), threads=threads)


def exact_dedup(texts: Any, *, normalize: Any = False, threads: int | None = None) -> Any:
    """Keep mask of first copies (see ``dedup.pyi``)."""
    return _native._exact_dedup(texts, _normalize_arg(normalize), threads=threads)


class MinHashDedup:
    """An in-memory MinHash index with datatrove's decisions (see ``dedup.pyi``)."""

    __slots__ = ("_config", "_language", "_python", "_state", "_tokenizer")

    def __init__(
        self,
        config: Any = None,
        *,
        threads: int | None = None,
        language: Any = "eng",
        python_tokenizer: bool = False,
        **kwargs: Any,
    ) -> None:
        self._config = _minhash_config(config, kwargs)
        src = _words_source(language, python_tokenizer, "pyroid.dedup.MinHashDedup")
        self._language = language
        self._python = bool(python_tokenizer)
        self._tokenizer = None if isinstance(src, str) else src
        native = _native_config(self._config, src if isinstance(src, str) else None)
        self._state = _native._MinHashState(native, threads=threads)

    @property
    def config(self) -> MinhashConfig:
        """The configuration (a :class:`MinhashConfig`)."""
        return self._config

    @property
    def language(self) -> Any:
        """The ``language`` documents are tokenized for (a code, or a tokenizer object)."""
        return self._language

    def add(self, texts: Any, *, threads: int | None = None) -> range:
        """Add a batch of documents; returns the indices they were given."""
        if self._tokenizer is not None:
            tokens, _, _ = _python_tokens(
                texts, self._config, self._tokenizer, threads, "pyroid.dedup.MinHashDedup.add"
            )
            start, stop = self._state._add_tokens(tokens, threads=threads)
        else:
            start, stop = self._state.add(texts, threads=threads)
        return range(start, stop)

    def fit(self, texts: Any, *, threads: int | None = None) -> Any:
        """Replace the index with ``texts`` and return their keep mask."""
        if self._tokenizer is not None:
            tokens, shape, docs = _python_tokens(
                texts, self._config, self._tokenizer, threads, "pyroid.dedup.MinHashDedup.fit"
            )
            return _mask(shape, docs, self._state._fit_tokens(tokens, threads=threads), threads)
        return self._state.fit(texts, threads=threads)

    def keep_mask(self) -> list[bool]:
        """``True`` for each document to keep, over every document added."""
        return self._state.keep_mask()  # type: ignore[no-any-return]

    def duplicates(self) -> list[int]:
        """Indices of the documents to remove, ascending."""
        return self._state.duplicates()  # type: ignore[no-any-return]

    def clusters(self) -> list[list[int]]:
        """Clusters of near-duplicates: the kept document first, then the removed ones."""
        return self._state.clusters()  # type: ignore[no-any-return]

    def cluster_ids(self) -> list[int]:
        """Per-document ``minhash_cluster_id``, as datatrove writes it (-1: in no cluster)."""
        return self._state.cluster_ids()  # type: ignore[no-any-return]

    def cluster_sizes(self) -> list[int]:
        """Per-document ``minhash_cluster_size``, as datatrove writes it (1: in no cluster)."""
        return self._state.cluster_sizes()  # type: ignore[no-any-return]

    def representatives(self) -> list[int]:
        """For each document, the index of the document kept in its place."""
        return self._state.representatives()  # type: ignore[no-any-return]

    def signature(self, index: int) -> list[int] | None:
        """The stored signature of document ``index`` (``None``: no shingles)."""
        return self._state.signature(index)  # type: ignore[no-any-return]

    def reset(self) -> None:
        """Forget every document."""
        self._state.reset()

    def __len__(self) -> int:
        return len(self._state)

    def __repr__(self) -> str:
        lang = "" if self._language == "eng" else f", language={self._language!r}"
        python = ", python_tokenizer=True" if self._python else ""
        return f"MinHashDedup({self._config!r}{lang}{python}, documents={len(self)})"


class ExactDedup:
    """Streaming exact dedup: keep the first copy of each text (see ``dedup.pyi``)."""

    __slots__ = ("_normalize", "_state")

    def __init__(self, *, normalize: Any = False) -> None:
        self._normalize = normalize
        self._state = _native._ExactState(_normalize_arg(normalize))

    def add(self, texts: Any, *, threads: int | None = None) -> Any:
        """Keep mask for one more batch: ``True`` for each first copy so far."""
        return self._state.add(texts, threads=threads)

    def reset(self) -> None:
        """Forget every text seen."""
        self._state.reset()

    def __len__(self) -> int:
        return len(self._state)

    def __repr__(self) -> str:
        return f"ExactDedup(normalize={self._normalize!r}, distinct={len(self)})"


# ======================================================================================
# More datatrove 0.10.1 steps: one section each (each extends __all__).
# ======================================================================================

# ---- SentenceDedup ----
# datatrove's SentenceDedupSignature -> SentenceFindDedups -> SentenceDedupFilter, in memory.

from typing import Generic, NamedTuple, TypeVar  # noqa: E402

__all__ += ["SentDedupConfig", "SentenceDedup", "SentenceDedupResult", "sentence_dedup"]


@dataclass(frozen=True)
class SentDedupConfig:
    """datatrove's ``SentDedupConfig``, with the same fields and defaults."""

    n_sentences: int = 3
    split_sentences: bool = True
    only_dedup_in_index: bool = True
    min_doc_words: int = 50
    min_num_sentences: int = 3
    min_words_to_remove_span: int = 0
    norm_config: TextNormConfig = field(default_factory=TextNormConfig)
    hash_config: HashConfig = field(default_factory=HashConfig)


_SentT = TypeVar("_SentT")
_SentM = TypeVar("_SentM")


class SentenceDedupResult(NamedTuple, Generic[_SentT, _SentM]):
    """What :func:`sentence_dedup` and :meth:`SentenceDedup.filter` return: ``(text, keep)``."""

    text: _SentT
    keep: _SentM


_SENT_FIELDS = tuple(f.name for f in fields(SentDedupConfig))
_SENT_INT_MAX = 2**63 - 1


def _threshold(name: str, value: object) -> int:
    """A threshold that datatrove disables for any value ``<= 0``."""
    return _int(name, value, -(2**63), _SENT_INT_MAX)


def _sent_config(config: object, kwargs: dict[str, Any]) -> SentDedupConfig:
    """``config`` (pyroid's, datatrove's, a dict or None) with ``kwargs`` applied.

    ``kwargs`` are ``SentDedupConfig`` fields, plus ``hash_fc`` / ``precision`` as
    shorthands for ``hash_config``.
    """
    if config is None:
        cfg = SentDedupConfig()
    elif isinstance(config, SentDedupConfig):
        cfg = config
    elif isinstance(config, dict):
        kwargs = {**config, **kwargs}
        cfg = SentDedupConfig()
    else:
        cfg = _from_fields(SentDedupConfig, config, _SENT_FIELDS)
    if kwargs:
        hash_kw = {k: kwargs.pop(k) for k in _HASH_FIELDS if k in kwargs}
        unknown = set(kwargs) - set(_SENT_FIELDS)
        if unknown:
            raise TypeError(f"unknown SentDedupConfig fields: {sorted(unknown)}")
        cfg = replace(cfg, **kwargs)
        if hash_kw:
            cfg = replace(cfg, hash_config=replace(_hash_config(cfg.hash_config), **hash_kw))
    return SentDedupConfig(
        n_sentences=_int("n_sentences", cfg.n_sentences, 1, 2**31),
        split_sentences=bool(cfg.split_sentences),
        only_dedup_in_index=bool(cfg.only_dedup_in_index),
        min_doc_words=_threshold("min_doc_words", cfg.min_doc_words),
        min_num_sentences=_threshold("min_num_sentences", cfg.min_num_sentences),
        min_words_to_remove_span=_threshold(
            "min_words_to_remove_span", cfg.min_words_to_remove_span
        ),
        norm_config=_norm_config(cfg.norm_config),
        hash_config=_hash_config(cfg.hash_config),
    )


def _sent_language(language: object) -> str:
    if not isinstance(language, str):
        raise TypeError(f"language must be a str, not {type(language).__name__}")
    return language


@lru_cache(maxsize=64)
def _native_sent(cfg: SentDedupConfig, language: str) -> Any:
    return _native._SentenceConfig(
        n_sentences=cfg.n_sentences,
        split_sentences=cfg.split_sentences,
        only_dedup_in_index=cfg.only_dedup_in_index,
        min_doc_words=max(cfg.min_doc_words, 0),
        min_num_sentences=max(cfg.min_num_sentences, 0),
        min_words_to_remove_span=max(cfg.min_words_to_remove_span, 0),
        hash_fc=cfg.hash_config.hash_fc,
        precision=cfg.hash_config.precision,
        norm=_native_norm(cfg.norm_config),
        language=language,
    )


def _no_tokenizer(language: str) -> Exception:
    """Datatrove's ValueError for an unknown code; else why pyroid can't tokenize it."""
    try:
        cls, arg, _ = _filters._backend(language)
    except ValueError as e:
        return e
    return NotImplementedError(
        f"pyroid has no native word tokenizer for language {language!r}: datatrove tokenizes it "
        f"with {_filters._describe(cls, arg)}, and pyroid's sentence dedup has no Python-tokenizer "
        "path; run datatrove's steps (pyroid.integrations.datatrove does, with a FallbackWarning). "
        "pyroid.filters.languages() lists the native codes; split_sentences=False with "
        "min_doc_words, min_num_sentences and min_words_to_remove_span <= 0 needs no tokenizer"
    )


class SentenceDedup:
    """datatrove's sentence-level dedup, in memory (see ``dedup.pyi``)."""

    __slots__ = ("_config", "_language", "_native", "_state", "_threads")

    def __init__(
        self,
        config: Any = None,
        *,
        language: str = "eng",
        threads: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._config = _sent_config(config, kwargs)
        self._language = _sent_language(language)
        self._native = _native_sent(self._config, self._language)
        if not self._native.has_tokenizer:
            err = _no_tokenizer(self._language)
            # datatrove loads the tokenizer in __init__: an unknown code always fails.
            if isinstance(err, ValueError) or self._native.needs_tokenizer(True):
                raise err
        self._threads = threads
        self._state = _native._SentenceState(self._native, threads=threads)

    @property
    def config(self) -> SentDedupConfig:
        """The configuration (a :class:`SentDedupConfig`)."""
        return self._config

    @property
    def language(self) -> str:
        """The datatrove language code of the word tokenizer."""
        return self._language

    def add(self, texts: Any, *, threads: int | None = None) -> range:
        """Stage 1 for a batch; returns the indices the documents were given."""
        start, stop = self._state.add(texts, threads=threads)
        return range(start, stop)

    def duplicates(self) -> list[tuple[int, int]]:
        """``(document, window)`` pairs to remove, sorted (stage 2's output)."""
        return self._state.duplicates().pairs()  # type: ignore[no-any-return]

    def filter(
        self, texts: Any, *, start: int = 0, threads: int | None = None
    ) -> SentenceDedupResult[Any, Any]:
        """Stage 3 for documents ``start``, ``start + 1``, ... (see ``dedup.pyi``)."""
        first = _int("start", start, 0, _SENT_INT_MAX)
        text, keep, _, _ = self._state.duplicates().filter(
            texts,
            self._native,
            start=first,
            threads=self._threads if threads is None else threads,
        )
        return SentenceDedupResult(text, keep)

    def fit(self, texts: Any, *, threads: int | None = None) -> SentenceDedupResult[Any, Any]:
        """Replace the documents with ``texts`` and run the three stages on them."""
        text, keep = self._state.fit(texts, threads=threads)
        return SentenceDedupResult(text, keep)

    def reset(self) -> None:
        """Forget every document."""
        self._state.reset()

    def __len__(self) -> int:
        return len(self._state)

    def __repr__(self) -> str:
        return (
            f"SentenceDedup({self._config!r}, language={self._language!r}, documents={len(self)})"
        )


def sentence_dedup(
    texts: Any,
    config: Any = None,
    *,
    language: str = "eng",
    threads: int | None = None,
    **kwargs: Any,
) -> SentenceDedupResult[Any, Any]:
    """The three sentence-dedup stages of datatrove over one corpus (see ``dedup.pyi``)."""
    return SentenceDedup(config, language=language, threads=threads, **kwargs).fit(texts)


# ---- end SentenceDedup ----
