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

from pyroid import _versions
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

_versions.check(__name__, "datatrove", f"datatrove=={DATATROVE_VERSION}")


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
    """datatrove's ``HashConfig``: the shingle hash function and its width.

    Attributes:
        precision: 64 or 32 bits.
        hash_fc: ``"xxhash"`` (xxHash64/xxHash32, seed 0) or ``"sha1"`` (the
            first 8 or 4 bytes of SHA-1, little-endian).
    """

    precision: int = 64
    hash_fc: str = "xxhash"


@dataclass(frozen=True)
class TextNormConfig:
    """datatrove's ``TextNormConfig``: what :func:`simplify_text` does, in order.

    Attributes:
        lowercase: ``str.lower()``.
        norm_numbers: Every run of digits (any script, with at most one
            ``.``/``,``-style separator inside) becomes ``0``.
        norm_weekdays: English weekday names become ``WEEKDAY``.
        norm_monthnames: English month names become ``MONTH``.
        remove_punctuation: datatrove's punctuation set becomes spaces.
        norm_whitespace: Strip, and collapse whitespace runs to one space.
        norm_unicode_diacritics: NFD, then drop combining marks (``Mn``).
    """

    lowercase: bool = True
    norm_whitespace: bool = True
    remove_punctuation: bool = True
    norm_unicode_diacritics: bool = True
    norm_numbers: bool = True
    norm_weekdays: bool = False
    norm_monthnames: bool = False


@dataclass(frozen=True)
class MinhashConfig:
    """datatrove's ``MinhashConfig``, with the same fields and defaults.

    Attributes:
        n_grams: Words per shingle.
        num_buckets: Bands of the LSH index.
        hashes_per_bucket: Hashes per band; a pair needs all of them equal.
        seed: Seed of numpy's ``RandomState`` that draws the permutations
            (0 to 2**32 - 1).
        norm_config: The :class:`TextNormConfig` applied before shingling.
        hash_config: The :class:`HashConfig` of the shingle hash.
    """

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
    """MinHash signature of each document, identical to datatrove's.

    Each signature is ``num_buckets * hashes_per_bucket`` ints, bucket by
    bucket (datatrove's ``get_signature`` flattened); ``None`` for a document
    with fewer than ``n_grams`` words, which datatrove gives no signature.
    Arrow nulls give ``None`` too.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        config: A :class:`MinhashConfig`, datatrove's ``MinhashConfig``, a
            dict of its fields, or ``None`` for the defaults.
        threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at most
            ``n`` tasks at once; ``1`` runs on the calling thread.
        language: datatrove's ``MinhashDedupSignature(language=...)``: the
            language code whose word tokenizer splits the simplified text (see
            "Languages" above), or a word tokenizer object.
        python_tokenizer: Tokenize with datatrove's own tokenizer, in Python
            on the calling thread (needed for the languages pyroid does not
            tokenize natively; emits :class:`pyroid.FallbackWarning` once per
            function and language). ``simplify_text``, shingling and MinHash
            still run in Rust.
        **kwargs: ``MinhashConfig`` fields to override, plus ``hash_fc`` and
            ``precision``.

    Returns:
        One signature (str input) or a list with one entry per document.

    Raises:
        pyroid.InputError: ``texts`` is not one of the accepted kinds.
        TypeError: An unknown or mistyped config field, or a ``language``
            that is neither a str nor a word tokenizer.
        ValueError: A config value out of range, or a ``language`` datatrove
            has no tokenizer for (datatrove's message).
        NotImplementedError: pyroid does not tokenize ``language`` natively
            and ``python_tokenizer`` is not set.
        pyroid.PyroidError: The Python-tokenizer path needs datatrove or the
            tokenizer's package, and it is not installed.
        UnicodeEncodeError: A str contains a lone surrogate.
        UnicodeDecodeError: An Arrow row is not valid UTF-8.
    """
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
    """Keep mask of datatrove's MinHash dedup over ``texts`` (one corpus).

    ``minhash_dedup(texts, cfg)`` is ``MinHashDedup(cfg).fit(texts)``: ``False``
    exactly for the documents datatrove's four stages remove.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        config: As for :func:`minhash_signatures`.
        threads: As for :func:`minhash_signatures`.
        language: As for :func:`minhash_signatures`.
        python_tokenizer: As for :func:`minhash_signatures`.
        **kwargs: As for :func:`minhash_signatures`.

    Returns:
        A mask in the same kind as ``texts``: bool, list, tuple, or a
        :class:`pyroid.Column` of bools (Arrow nulls stay null).

    Raises the same exceptions as :func:`minhash_signatures`.
    """
    cfg = _minhash_config(config, kwargs)
    src = _words_source(language, python_tokenizer, "pyroid.dedup.minhash_dedup")
    if isinstance(src, str):
        return _native._MinHashState(_native_config(cfg, src), threads=threads).fit(texts)
    state = _native._MinHashState(_native_config(cfg), threads=threads)
    tokens, shape, docs = _python_tokens(texts, cfg, src, threads, "pyroid.dedup.minhash_dedup")
    return _mask(shape, docs, state._fit_tokens(tokens), threads)


def simplify_text(texts: Any, config: Any = None, *, threads: int | None = None) -> Any:
    """Normalise text with datatrove's ``simplify_text`` (what MinHash shingles see).

    Byte-identical to ``datatrove.utils.text.simplify_text`` on the same
    CPython version (its ``unicodedata``) with the ``regex`` module's digit
    class. Unchanged documents are returned as the original objects.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        config: A :class:`TextNormConfig`, datatrove's ``TextNormConfig``, a
            dict of its fields, or ``None`` for the defaults.
        threads: As for :func:`minhash_signatures`.

    Returns:
        The same kind as ``texts``: str, list, tuple, or a :class:`pyroid.Column`
        of strings.

    Raises:
        pyroid.InputError: ``texts`` is not one of the accepted kinds.
        UnicodeEncodeError: A str contains a lone surrogate.
        UnicodeDecodeError: An Arrow row is not valid UTF-8.
    """
    return _native._simplify_text(texts, _native_norm(_norm_config(config)), threads=threads)


def exact_dedup(texts: Any, *, normalize: Any = False, threads: int | None = None) -> Any:
    """Keep mask of exact duplicates: ``True`` for the first copy of each text.

    Texts are compared by a 128-bit XXH3 hash (for 10⁹ documents the chance
    of any false match is about 10⁻²¹). Arrow nulls stay null and match
    nothing.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        normalize: ``False`` compares the raw text; ``True`` compares
            :func:`simplify_text` with the default :class:`TextNormConfig`;
            a ``TextNormConfig`` (pyroid's or datatrove's) or dict uses that.
        threads: As for :func:`minhash_signatures`.

    Returns:
        A mask in the same kind as ``texts``.

    Raises:
        pyroid.InputError: ``texts`` is not one of the accepted kinds.
        UnicodeEncodeError: A str contains a lone surrogate.
        UnicodeDecodeError: An Arrow row is not valid UTF-8.
    """
    return _native._exact_dedup(texts, _normalize_arg(normalize), threads=threads)


class MinHashDedup:
    """An in-memory MinHash index that makes datatrove's removals.

    Add documents in batches with :meth:`add` (only their signatures are
    kept, ``8 * num_buckets * hashes_per_bucket`` bytes each, in Rust), then
    read the decisions for everything added so far. Decisions are global:
    a later batch can change which document of a cluster is kept, so masks
    are only final once every batch is in. :meth:`fit` does one corpus in a
    single call. Methods may be called from several threads.

    Example::

        d = MinHashDedup(hash_fc="sha1")       # FineWeb's configuration
        for batch in batches:
            d.add(batch)
        keep = d.keep_mask()                   # one bool per document added

    Args:
        config: As for :func:`minhash_signatures`.
        threads: Default ``threads=`` for :meth:`add`, :meth:`fit` and the
            clustering.
        language: As for :func:`minhash_signatures`, for every document added.
        python_tokenizer: As for :func:`minhash_signatures` (the warning is
            emitted once, when the index is made).
        **kwargs: As for :func:`minhash_signatures`.
    """

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
        """The ``language`` the documents are tokenized for (a code or a tokenizer object)."""
        return self._language

    def add(self, texts: Any, *, threads: int | None = None) -> range:
        """Add a batch of documents.

        Returns:
            The indices given to the batch's documents (Arrow nulls get one
            too, and are always kept).
        """
        if self._tokenizer is not None:
            tokens, _, _ = _python_tokens(
                texts, self._config, self._tokenizer, threads, "pyroid.dedup.MinHashDedup.add"
            )
            start, stop = self._state._add_tokens(tokens, threads=threads)
        else:
            start, stop = self._state.add(texts, threads=threads)
        return range(start, stop)

    def fit(self, texts: Any, *, threads: int | None = None) -> Any:
        """Replace the index with ``texts`` and return their keep mask.

        Returns:
            A mask in the same kind as ``texts`` (see :func:`minhash_dedup`).
        """
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
        """Indices of the documents to remove, ascending (datatrove's ``.remove`` file)."""
        return self._state.duplicates()  # type: ignore[no-any-return]

    def clusters(self) -> list[list[int]]:
        """Every cluster of two or more near-duplicates.

        Each cluster lists the kept document first, then the removed ones in
        ascending order. Clusters are ordered by their smallest member, so
        ``clusters()[k]`` is datatrove's cluster id ``k``.
        """
        return self._state.clusters()  # type: ignore[no-any-return]

    def cluster_ids(self) -> list[int]:
        """Per-document ``minhash_cluster_id``, as datatrove writes it (``-1``: in no cluster)."""
        return self._state.cluster_ids()  # type: ignore[no-any-return]

    def cluster_sizes(self) -> list[int]:
        """Per-document ``minhash_cluster_size``, as datatrove writes it (``1``: no cluster)."""
        return self._state.cluster_sizes()  # type: ignore[no-any-return]

    def representatives(self) -> list[int]:
        """For each document, the index of its cluster's kept document (itself if kept)."""
        return self._state.representatives()  # type: ignore[no-any-return]

    def signature(self, index: int) -> list[int] | None:
        """The stored signature of document ``index`` (``None``: no shingles).

        Raises:
            IndexError: ``index`` is not a document added so far.
        """
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
    """Streaming exact dedup: keep the first copy of each text, across batches.

    Holds one 128-bit hash per distinct text (see :func:`exact_dedup`), so
    each batch's mask is final as soon as :meth:`add` returns.

    Args:
        normalize: As for :func:`exact_dedup`.
    """

    __slots__ = ("_normalize", "_state")

    def __init__(self, *, normalize: Any = False) -> None:
        self._normalize = normalize
        self._state = _native._ExactState(_normalize_arg(normalize))

    def add(self, texts: Any, *, threads: int | None = None) -> Any:
        """Keep mask for one more batch: ``True`` for each first copy so far.

        Returns:
            A mask in the same kind as ``texts`` (Arrow nulls stay null).
        """
        return self._state.add(texts, threads=threads)

    def reset(self) -> None:
        """Forget every text seen."""
        self._state.reset()

    def __len__(self) -> int:
        """Number of distinct texts seen."""
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
    """datatrove's ``SentDedupConfig``, with the same fields and defaults.

    Attributes:
        n_sentences: Sentences per window (C4's three); at least 1.
        split_sentences: ``True``: the language tokenizer's sentences (datatrove's
            ``span_tokenize``, ported); ``False``: lines (``str.splitlines()``).
        only_dedup_in_index: With index files (the datatrove drop-ins only),
            remove only the windows found in the index. No effect in memory.
        min_doc_words: A document whose text changed is dropped if it has
            fewer words (``<= 0``: no minimum).
        min_num_sentences: A document whose text changed is dropped if it has
            fewer sentences (``<= 0``: no minimum).
        min_words_to_remove_span: Restore a removed run whose first sentence
            has fewer words, as datatrove does: only that first sentence comes
            back (``<= 0``: never).
        norm_config: The :class:`TextNormConfig` applied to each sentence
            before hashing.
        hash_config: The :class:`HashConfig` of the window hash.
    """

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
    """What :func:`sentence_dedup` and the :class:`SentenceDedup` methods return.

    Both fields are in the caller's container kind (str/bool, list, tuple, or a
    :class:`pyroid.Column`; Arrow nulls stay null).
    """

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
        + _filters._DATA_HINTS.get((cls, arg), "").replace(" Or", "; or").rstrip(".")
    )


class SentenceDedup:
    """datatrove's sentence-level dedup, in memory and multi-threaded.

    C4's rule ("we discarded all but one of any three-sentence span occurring
    more than once"), with the same removals and keep decisions as datatrove
    0.10.1's ``SentenceDedupSignature`` -> ``SentenceFindDedups`` ->
    ``SentenceDedupFilter`` run as one task:

    1. *Signatures* (:meth:`add`). A document's sentences (datatrove's
       ``sent_tokenize``: spaCy's sentences, stripped, empty ones dropped; or,
       with ``split_sentences=False``, its lines) each go through
       :func:`simplify_text`; every window of ``n_sentences`` consecutive
       ones, joined by a space, is hashed (``hash_fc``) unless it is blank.
       A document with fewer than ``n_sentences`` sentences has none.
    2. *Duplicates* (:meth:`duplicates`). For each hash seen more than once,
       every window but the first (lowest document, then window) is removed.
    3. *Filter* (:meth:`filter`). The ``n_sentences`` sentences of each
       removed window are cut out. Sentences are now spaCy's
       ``span_tokenize`` spans, each with the whitespace before it, which
       (unlike step 1) include a whitespace-only last sentence; the result is
       left-stripped (lines are rejoined with newlines). A document whose
       text changed is dropped if it has fewer than ``min_doc_words`` words
       or ``min_num_sentences`` sentences; an empty text is always dropped,
       even with nothing removed, as in datatrove.

    datatrove's quirks are kept: ``min_words_to_remove_span`` only counts,
    and only restores, the *first* sentence of a removed run.

    Documents are numbered in the order they are added, as datatrove numbers
    the documents of one task. Add every batch first, then filter them (pass
    ``start=`` for batches after the first); :meth:`fit` does one corpus in a
    single call. Only the signatures are kept (16 bytes per window, in Rust).
    Methods may be called from several threads.

    Example::

        d = SentenceDedup(min_doc_words=50)          # datatrove's example config
        ranges = [d.add(batch) for batch in batches]
        for batch, r in zip(batches, ranges):
            texts, keep = d.filter(batch, start=r.start)

    Known differences from datatrove:

    - A str with a lone surrogate raises ``UnicodeEncodeError`` (so does
      datatrove, except with ``split_sentences=False`` on a document with
      fewer than ``n_sentences`` lines).
    - ``n_sentences`` below 1 raises ``ValueError``.
    - A hashed window index past 65,535 raises datatrove's ``OverflowError``
      (numpy's ``<u2``) when its batch is added, and the batch is not added
      (datatrove fails when it saves the task's signatures).
    - Index files: use the datatrove drop-ins
      (``pyroid.integrations.datatrove``).

    Args:
        config: As for :func:`sentence_dedup`.
        language: As for :func:`sentence_dedup`; any code is accepted when no
            tokenizer is needed (``split_sentences=False`` and the three
            word/sentence thresholds ``<= 0``).
        threads: Default ``threads=`` for :meth:`add`, :meth:`filter` and
            :meth:`fit`.
        **kwargs: As for :func:`sentence_dedup`.
    """

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
        """Stage 1 for a batch of documents.

        Returns:
            The indices given to the batch's documents (Arrow nulls get one,
            and no signatures).

        Raises:
            OverflowError: A window index past 65,535 (nothing is added).
        """
        start, stop = self._state.add(texts, threads=threads)
        return range(start, stop)

    def duplicates(self) -> list[tuple[int, int]]:
        """Stage 2: the ``(document, window)`` pairs to remove, sorted.

        These are the records of datatrove's ``.c4_dup`` files for the task.
        """
        return self._state.duplicates().pairs()  # type: ignore[no-any-return]

    def filter(
        self, texts: Any, *, start: int = 0, threads: int | None = None
    ) -> SentenceDedupResult[Any, Any]:
        """Stage 3 for documents ``start``, ``start + 1``, ... (as added).

        Pass the same documents, in the same order, as to :meth:`add`.
        Decisions are global, so filter once every batch has been added.

        Returns:
            A :class:`SentenceDedupResult` ``(text, keep)`` in the caller's
            kind.
        """
        first = _int("start", start, 0, _SENT_INT_MAX)
        text, keep, _, _ = self._state.duplicates().filter(
            texts,
            self._native,
            start=first,
            threads=self._threads if threads is None else threads,
        )
        return SentenceDedupResult(text, keep)

    def fit(self, texts: Any, *, threads: int | None = None) -> SentenceDedupResult[Any, Any]:
        """Replace the documents with ``texts`` and run the three stages on them.

        Returns:
            A :class:`SentenceDedupResult` ``(text, keep)`` (see
            :func:`sentence_dedup`).
        """
        text, keep = self._state.fit(texts, threads=threads)
        return SentenceDedupResult(text, keep)

    def reset(self) -> None:
        """Forget every document."""
        self._state.reset()

    def __len__(self) -> int:
        """Number of documents added."""
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
    """The three sentence-dedup stages of datatrove over one corpus.

    ``sentence_dedup(texts, cfg)`` is ``SentenceDedup(cfg).fit(texts)``: the
    texts and keep decisions of datatrove's ``SentenceDedupSignature`` ->
    ``SentenceFindDedups`` -> ``SentenceDedupFilter`` run as one task (see
    :class:`SentenceDedup`). The input is read once.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        config: A :class:`SentDedupConfig`, datatrove's ``SentDedupConfig``, a
            dict of its fields, or ``None`` for the defaults.
        language: datatrove's language code (as its ``language=``), for the sentence and
            word tokenizer: any code ``pyroid.filters.languages()`` marks native
            (spaCy rule-based languages, Chinese with jieba, Vietnamese with pyvi,
            Japanese with an installed SudachiDict, Thai with an installed
            PyThaiNLP, the whitespace and IndicNLP tokenizers). Codes whose
            datatrove tokenizer pyroid does not port (Stanza, Kiwi, ...) raise
            ``NotImplementedError``; use the datatrove drop-ins.
        threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at most
            ``n`` tasks at once; ``1`` runs on the calling thread.
        **kwargs: ``SentDedupConfig`` fields to override, plus ``hash_fc`` and
            ``precision``.

    Returns:
        A :class:`SentenceDedupResult` ``(text, keep)``.

    Raises:
        pyroid.InputError: ``texts`` is not one of the accepted kinds.
        TypeError: An unknown or mistyped config field.
        ValueError: A config value out of range, ``language`` is not a datatrove
            code (datatrove's message), or a text of 2**30 characters or more
            (2**32 - 64 for Thai).
        NotImplementedError: No native word tokenizer for ``language``.
        OverflowError: A document has more than 65,536 sentence windows
            (datatrove's ``<u2`` window index).
        UnicodeEncodeError: A str contains a lone surrogate.
        UnicodeDecodeError: An Arrow row is not valid UTF-8.
    """
    return SentenceDedup(config, language=language, threads=threads, **kwargs).fit(texts)


# ---- end SentenceDedup ----
