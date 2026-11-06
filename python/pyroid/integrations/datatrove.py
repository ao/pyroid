# Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.
# mypy: disallow-subclassing-any=False
# (datatrove ships no type information: its classes are Any to mypy.)
"""Drop-in datatrove pipeline steps backed by pyroid.

Swap the import, keep the pipeline::

    # from datatrove.pipeline.filters import GopherQualityFilter, LanguageFilter
    from pyroid.integrations.datatrove import GopherQualityFilter, LanguageFilter

Each class here subclasses datatrove's own step of the same name and takes the
same constructor arguments, plus the keyword-only ``batch_size`` (documents per
pyroid call, default 1000) and ``threads`` (pyroid's ``threads=``, default 1:
datatrove already runs one task per process; pass ``None`` for pyroid's whole
thread pool when you run few tasks). The documents that come out are the same
as with datatrove's step: same ``text``, same ``metadata``, same keep/drop
decisions and drop reasons, so exclusion writers and later steps see the same
data. The work happens in Rust, in batches, without the GIL:

- :class:`FTFYFormatter`: ``pyroid.ftfy`` (byte-identical to ftfy 6.3.1; ftfy
  itself is not needed).
- :class:`LanguageFilter`: datatrove's own ``filter()`` and ``FastTextLID.predict``
  code, with the fastText call replaced by one batched ``pyroid.fasttext``
  prediction (no ``fasttext-numpy2-wheel`` or ``fasteners`` needed when
  ``model_path`` is given; without it the model is downloaded to datatrove's
  own cache, as datatrove does).
- :class:`GopherRepetitionFilter`, :class:`GopherQualityFilter`,
  :class:`C4QualityFilter`, :class:`FineWebQualityFilter`: ``pyroid.filters``
  (spaCy is not needed).
- :class:`MinhashDedupSignature`: ``pyroid.dedup`` signatures, written to the
  same files, byte for byte, so datatrove's ``MinhashDedupBuckets`` ->
  ``MinhashDedupCluster`` -> ``MinhashDedupFilter`` run on them unchanged.

What differs: per-line statistics of ``C4QualityFilter`` (``line-total``,
``line-filter-*``) are not counted, and batched steps also count ``batches``.
A language pyroid cannot tokenize (anything but the English codes in
``pyroid.filters.english_languages()``) runs datatrove's own code, with a
one-time :class:`pyroid.FallbackWarning`.

Needs datatrove (and numpy for ``LanguageFilter``/``MinhashDedupSignature``):
``pip install 'pyroid[datatrove]'``.
"""

from __future__ import annotations

import inspect
import warnings
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any, cast

from datatrove.pipeline.dedup import minhash as _dt_minhash  # type: ignore[import-not-found,import-untyped,unused-ignore]
from datatrove.pipeline.filters import c4_filters as _dt_c4  # type: ignore[import-not-found,import-untyped,unused-ignore]
from datatrove.pipeline.filters import fineweb_quality_filter as _dt_fineweb  # type: ignore[import-not-found,import-untyped,unused-ignore]
from datatrove.pipeline.filters import gopher_quality_filter as _dt_gq  # type: ignore[import-not-found,import-untyped,unused-ignore]
from datatrove.pipeline.filters import gopher_repetition_filter as _dt_gr  # type: ignore[import-not-found,import-untyped,unused-ignore]
from datatrove.pipeline.filters import language_filter as _dt_lang  # type: ignore[import-not-found,import-untyped,unused-ignore]
from datatrove.pipeline.formatters import ftfy as _dt_ftfy  # type: ignore[import-not-found,import-untyped,unused-ignore]
from datatrove.utils import lid as _dt_lid  # type: ignore[import-not-found,import-untyped,unused-ignore]
from datatrove.utils._import_utils import check_required_dependencies  # type: ignore[import-not-found,import-untyped,unused-ignore]
from datatrove.utils.batching import batched  # type: ignore[import-not-found,import-untyped,unused-ignore]
from datatrove.utils.typeshelper import StatHints  # type: ignore[import-not-found,import-untyped,unused-ignore]

import pyroid
import pyroid.ftfy

__all__ = [
    "C4QualityFilter",
    "FTFYFormatter",
    "FineWebQualityFilter",
    "GopherQualityFilter",
    "GopherRepetitionFilter",
    "LanguageFilter",
    "MinhashDedupSignature",
]

DEFAULT_BATCH_SIZE = 1000
_WARNED: set[str] = set()


def _check_datatrove_version() -> None:
    from importlib.metadata import PackageNotFoundError, version

    import pyroid.filters

    try:
        installed = version("datatrove")
    except PackageNotFoundError:  # pragma: no cover - imported from a source tree
        return
    if installed != pyroid.filters.DATATROVE_VERSION:
        warnings.warn(
            f"pyroid.integrations.datatrove reproduces datatrove "
            f"{pyroid.filters.DATATROVE_VERSION}'s decisions; datatrove {installed} is installed, "
            "whose own steps may differ",
            RuntimeWarning,
            stacklevel=3,
        )


_check_datatrove_version()


def _fallback(step: str, reason: str) -> None:
    key = f"{step}:{reason}"
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(
            f"pyroid.integrations.datatrove.{step}: {reason} is not supported by pyroid; "
            "running datatrove's own code (slow path). This warning is shown once.",
            pyroid.FallbackWarning,
            stacklevel=3,
        )


def _batch_size(n: object) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"batch_size must be a positive int, not {n!r}")
    return n


_EXTRA = (
    inspect.Parameter("batch_size", inspect.Parameter.KEYWORD_ONLY, default=DEFAULT_BATCH_SIZE),
    inspect.Parameter("threads", inspect.Parameter.KEYWORD_ONLY, default=1),
)


def _signature_of(parent: type, *extra: inspect.Parameter) -> inspect.Signature:
    """``parent.__init__``'s signature plus our keyword-only extras (for help() and IDEs)."""
    sig = inspect.signature(cast(Any, parent).__init__)
    params = list(sig.parameters.values())
    tail = [p for p in params if p.kind is inspect.Parameter.VAR_KEYWORD]
    head = [p for p in params if p.kind is not inspect.Parameter.VAR_KEYWORD]
    return sig.replace(parameters=[*head, *extra, *tail])


class _Step:
    """Mixin: pyroid's own dependency check instead of the datatrove parent's.

    ``PipelineStep.__new__`` checks ``_requires_dependencies`` over the whole MRO,
    which would demand e.g. ftfy or ``fasttext-numpy2-wheel`` that we don't use.
    """

    _pyroid_requires: tuple[str, ...] = ()

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls._pyroid_requires:
            check_required_dependencies(cls.__name__, list(cls._pyroid_requires))
        return object.__new__(cls)


# ------------------------------------------------------------------------------------------- ftfy


class FTFYFormatter(_Step, _dt_ftfy.FTFYFormatter):
    """datatrove's ``FTFYFormatter`` (same arguments and defaults) on ``pyroid.ftfy``.

    ``doc.text`` becomes ``ftfy.fix_text(doc.text, config=self.config)``, byte for
    byte, computed ``batch_size`` documents at a time.
    """

    def __init__(
        self, *args: Any, batch_size: int = DEFAULT_BATCH_SIZE, threads: Any = 1, **kwargs: Any
    ):
        # datatrove's __init__ builds the config with ftfy's TextFixerConfig: build pyroid's
        # instead from the same arguments (bound against datatrove's own signature).
        bound = inspect.signature(_dt_ftfy.FTFYFormatter.__init__).bind(self, *args, **kwargs)
        bound.apply_defaults()
        options = {k: v for k, v in bound.arguments.items() if k != "self"}
        _dt_ftfy.BaseFormatter.__init__(self)
        self.config = pyroid.ftfy.TextFixerConfig(**options)
        self.batch_size = _batch_size(batch_size)
        self.threads = threads

    def format(self, text: str) -> str:
        """One document (datatrove's API); :meth:`run` batches instead."""
        return pyroid.ftfy.fix_text(text, config=self.config)

    def run(self, data: Iterable[Any], rank: int = 0, world_size: int = 1) -> Any:
        """Fix ``doc.text`` of every document, ``batch_size`` at a time."""
        for batch in batched(data, self.batch_size):
            with self.track_time("batch"):
                fixed = pyroid.ftfy.fix_text_batch(
                    [d.text for d in batch], self.config, threads=self.threads
                )
            for doc, text in zip(batch, fixed, strict=True):
                self.stat_update(StatHints.total)
                doc.text = text
                yield doc


FTFYFormatter.__init__.__signature__ = _signature_of(_dt_ftfy.FTFYFormatter, *_EXTRA)  # type: ignore[attr-defined]


# -------------------------------------------------------------------------------------------- LID


class _Primed:
    """Stands in for the fastText model inside datatrove's ``FastTextLID.predict``.

    :meth:`LanguageFilter.filter_batch` queues the batch's predictions (one
    batched pyroid call); datatrove's own ``predict()`` then pops them in order,
    so its label parsing and ``.item()`` calls run unchanged.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.queue: deque[tuple[Any, Any]] = deque()

    def predict(
        self, text: str, k: int = 1, threshold: float = 0.0, on_unicode_error: str = "strict"
    ) -> Any:
        if self.queue:
            return self.queue.popleft()
        return self.model.predict(text, k, threshold, on_unicode_error)


class _PyroidLID:
    """Mixin for datatrove's ``FastTextLID`` backends: the model is a pyroid one."""

    model_path: str | None
    _model: Any

    @property
    def model(self) -> Any:
        if self._model is None:
            import pyroid.fasttext

            path = self.model_path
            if path is None:
                from datatrove.io import cached_asset_path_or_download  # type: ignore[import-not-found,import-untyped,unused-ignore]

                path = cached_asset_path_or_download(
                    self.MODEL_URL,  # type: ignore[attr-defined]
                    namespace="lid",
                    subfolder=self.MODEL_SUBFOLDER,  # type: ignore[attr-defined]
                    desc="fast-text language identifier model",
                )
            self._model = _Primed(pyroid.fasttext.load_model(path))
        return self._model


class PyroidFT176LID(_PyroidLID, _dt_lid.FT176LID):
    """datatrove's ``FT176LID`` (lid.176.bin) on ``pyroid.fasttext``."""

    def __init__(self, languages: list[str] | None = None, k: int = -1, *, model_path: Any = None):
        super().__init__(languages, k)
        self.model_path = None if model_path is None else str(model_path)


class PyroidGlotLID(_PyroidLID, _dt_lid.GlotLID):
    """datatrove's ``GlotLID`` on ``pyroid.fasttext``."""

    def __init__(
        self,
        languages: list[str] | None = None,
        k: int = -1,
        version: str = "v3",
        *,
        model_path: Any = None,
    ):
        super().__init__(languages, k, version)
        self.model_path = None if model_path is None else str(model_path)


class LanguageFilter(_Step, _dt_lang.LanguageFilter):
    """datatrove's ``LanguageFilter`` (same arguments) on ``pyroid.fasttext``.

    Extra keyword-only arguments: ``model_path`` (a local ``lid.176.bin`` or GlotLID
    ``.bin``; default: download to datatrove's cache, as datatrove does),
    ``batch_size`` and ``threads``.

    The decision and the ``language`` / ``language_score`` (and
    ``language_script``, ``top_language_*_score``) metadata come from datatrove's
    own ``filter()`` code. When only the top label can matter (no
    ``keep_top_pairs_threshold``, and no languages, ``label_only``, or a
    threshold above 0.501, since fastText's probabilities sum to at most
    1 + 176e-5), pyroid predicts the top label only instead of all of them.
    """

    _pyroid_requires = ("numpy",)

    def __init__(
        self,
        *args: Any,
        model_path: Any = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        threads: Any = 1,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        cls = PyroidFT176LID if self.backend == "ft176" else PyroidGlotLID
        self.model = cls(self.languages, model_path=model_path)
        self.batch_size = _batch_size(batch_size)
        self.threads = threads

    def _k(self) -> int:
        top_only = self.keep_top_pairs_threshold == -1 and (
            not self.languages or self.label_only or self.language_threshold > 0.501
        )
        return 1 if top_only else self.model.k

    def filter(self, doc: Any) -> Any:
        """One document (datatrove's API); :meth:`filter_batch` batches instead."""
        return self.filter_batch([doc])[0]

    def filter_batch(self, batch: list[Any]) -> list[Any]:
        """Predict the batch in one pyroid call, then run datatrove's ``filter()`` per doc."""
        primed: _Primed = self.model.model
        labels, probs = primed.model.predict(
            [d.text.replace("\n", " ") for d in batch], k=self._k(), threads=self.threads
        )
        primed.queue.extend(zip(labels, probs, strict=True))
        try:
            return [_dt_lang.LanguageFilter.filter(self, d) for d in batch]
        finally:
            primed.queue.clear()


LanguageFilter.__init__.__signature__ = _signature_of(  # type: ignore[attr-defined]
    _dt_lang.LanguageFilter,
    inspect.Parameter("model_path", inspect.Parameter.KEYWORD_ONLY, default=None),
    *_EXTRA,
)


# ---------------------------------------------------------------------------------------- filters


class _QualityFilter(_Step):
    """Shared ``filter`` / ``filter_batch`` for the pyroid.filters-backed steps."""

    _datatrove: Any
    language: str
    batch_size: int
    threads: Any

    def _setup(self, batch_size: int, threads: Any) -> None:
        self.batch_size = _batch_size(batch_size)
        self.threads = threads

    def filter(self, doc: Any) -> Any:
        """One document (datatrove's API); :meth:`filter_batch` batches instead."""
        return self.filter_batch([doc])[0]

    def filter_batch(self, batch: list[Any]) -> list[Any]:
        """One ``pyroid.filters.apply`` call for the whole batch."""
        import pyroid.filters as pf

        if self.language not in _english():
            _fallback(type(self).__name__, f"language={self.language!r}")
            parent: Callable[[Any, Any], Any] = self._datatrove.filter
            return [parent(self, d) for d in batch]
        res = pf.apply([d.text for d in batch], cast(Any, self), threads=self.threads)
        out: list[Any] = []
        for doc, keep, reason, text in zip(batch, res.keep, res.reason, res.text, strict=True):
            if keep:
                if text is not doc.text and text != doc.text:
                    doc.text = text  # C4's line removal
                out.append(True)
            else:
                out.append((False, reason) if reason else False)
        return out


_ENGLISH: list[frozenset[str]] = []


def _english() -> frozenset[str]:
    if not _ENGLISH:
        import pyroid.filters as pf

        _ENGLISH.append(frozenset(pf.english_languages()))
    return _ENGLISH[0]


class GopherRepetitionFilter(_QualityFilter, _dt_gr.GopherRepetitionFilter):
    """datatrove's ``GopherRepetitionFilter`` (same arguments) on ``pyroid.filters``."""

    _datatrove = _dt_gr.GopherRepetitionFilter

    def __init__(
        self, *args: Any, batch_size: int = DEFAULT_BATCH_SIZE, threads: Any = 1, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self._setup(batch_size, threads)


class GopherQualityFilter(_QualityFilter, _dt_gq.GopherQualityFilter):
    """datatrove's ``GopherQualityFilter`` (same arguments) on ``pyroid.filters``."""

    _datatrove = _dt_gq.GopherQualityFilter

    def __init__(
        self, *args: Any, batch_size: int = DEFAULT_BATCH_SIZE, threads: Any = 1, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self._setup(batch_size, threads)


class C4QualityFilter(_QualityFilter, _dt_c4.C4QualityFilter):
    """datatrove's ``C4QualityFilter`` (same arguments) on ``pyroid.filters``.

    Kept documents get the same line removal; per-line statistics are not counted.
    """

    _datatrove = _dt_c4.C4QualityFilter

    def __init__(
        self, *args: Any, batch_size: int = DEFAULT_BATCH_SIZE, threads: Any = 1, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self._setup(batch_size, threads)


class FineWebQualityFilter(_QualityFilter, _dt_fineweb.FineWebQualityFilter):
    """datatrove's ``FineWebQualityFilter`` (same arguments) on ``pyroid.filters``."""

    _datatrove = _dt_fineweb.FineWebQualityFilter

    def __init__(
        self, *args: Any, batch_size: int = DEFAULT_BATCH_SIZE, threads: Any = 1, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self._setup(batch_size, threads)


for _cls in (GopherRepetitionFilter, GopherQualityFilter, C4QualityFilter, FineWebQualityFilter):
    cast(Any, _cls.__init__).__signature__ = _signature_of(_cls._datatrove, *_EXTRA)
del _cls


# ----------------------------------------------------------------------------------------- MinHash


class MinhashDedupSignature(_Step, _dt_minhash.MinhashDedupSignature):
    """datatrove's ``MinhashDedupSignature`` (stage 1) on ``pyroid.dedup``.

    Writes the same ``bucket_XXX/RANK.minhash.sig`` files, byte for byte (the
    sort pass is datatrove's own), so datatrove's later MinHash stages run on
    them unchanged. spaCy is not needed. Note that datatrove's sort pass logs
    "Found existing sig files ... Skipping sig writing step": pyroid has just
    written them.
    """

    _pyroid_requires = ("numpy",)

    def __init__(
        self, *args: Any, batch_size: int = DEFAULT_BATCH_SIZE, threads: Any = 1, **kwargs: Any
    ) -> None:
        try:
            super().__init__(*args, **kwargs)
        except ImportError:
            # datatrove loads its spaCy word tokenizer here; pyroid has its own, so a missing
            # spaCy only matters for languages pyroid can't tokenize.
            bound = inspect.signature(_dt_minhash.MinhashDedupSignature.__init__).bind(
                self, *args, **kwargs
            )
            bound.apply_defaults()
            if bound.arguments["language"] not in _english() or not hasattr(self, "_hash_func"):
                raise
            self.word_tokenizer = None
            self.skip_existing_sigs = bound.arguments["skip_existing_sigs"]
        self.batch_size = _batch_size(batch_size)
        self.threads = threads

    def _write_unsorted(self, data: Iterable[Any], rank: int) -> None:
        import numpy as np

        import pyroid.dedup

        cfg = self.config
        hpb, nb = cfg.hashes_per_bucket, cfg.num_buckets
        fmt = cfg.hash_config.struct_format
        dtype = np.dtype(
            [(f"field{i + 1}", f"<{fmt}") for i in range(hpb)] + [(f"field{hpb + 1}", "<I")]
        )
        buckets = [
            self.output_folder.open(f"bucket_{bi:03d}/{rank:05d}.minhash.sig", mode="wb")
            for bi in range(nb)
        ]
        try:
            doc_idx = 0
            for batch in batched(data, self.batch_size):
                sigs = pyroid.dedup.minhash_signatures(
                    [d.text for d in batch], cfg, threads=self.threads
                )
                for _ in batch:
                    self.stat_update(StatHints.total)
                idx = [doc_idx + i for i, s in enumerate(sigs) if s is not None]
                doc_idx += len(batch)
                if not idx:
                    continue
                hashes = np.array([s for s in sigs if s is not None], dtype=f"<{fmt}")
                hashes = hashes.reshape(len(idx), nb, hpb)
                for bi, f in enumerate(buckets):
                    rec = np.empty(len(idx), dtype=dtype)
                    for h in range(hpb):
                        rec[f"field{h + 1}"] = hashes[:, bi, h]
                    rec[f"field{hpb + 1}"] = idx
                    f.write(rec.tobytes())
        finally:
            for f in buckets:
                f.close()

    def run(self, data: Iterable[Any], rank: int = 0, world_size: int = 1) -> None:
        """Write this rank's signatures, then datatrove's own sort-and-verify pass."""
        if self.language not in _english():
            _fallback("MinhashDedupSignature", f"language={self.language!r}")
            return super().run(data, rank, world_size)  # type: ignore[no-any-return]
        with self.track_time():
            if not self.check_can_skip_sig_writing(rank)[0]:
                self._write_unsorted(data, rank)
        saved = self.skip_existing_sigs
        self.skip_existing_sigs = True  # datatrove's run() then only sorts and verifies
        try:
            return super().run(iter(()), rank, world_size)  # type: ignore[no-any-return]
        finally:
            self.skip_existing_sigs = saved


MinhashDedupSignature.__init__.__signature__ = _signature_of(  # type: ignore[attr-defined]
    _dt_minhash.MinhashDedupSignature, *_EXTRA
)


# ======================================================================================
# More datatrove 0.10.1 steps: one section each (each extends __all__).
# ======================================================================================

# ---- PIIFormatter ----
from datatrove.pipeline.formatters import pii as _dt_pii  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: E402

__all__ += ["PIIFormatter"]


class PIIFormatter(_Step, _dt_pii.PIIFormatter):
    """datatrove's ``PIIFormatter`` (same arguments and defaults) on ``pyroid.filters``.

    ``doc.text`` gets the same email and IP replacements, byte for byte, in the
    same rotation: the counters are datatrove's own (``emails_replacer._replace_i``
    and ``ip_replacer._replace_i``), and whether an address is replaced is asked
    of datatrove's own validator, once per distinct address and batch. The
    regexes run in Rust, ``batch_size`` documents at a time.
    """

    def __init__(
        self, *args: Any, batch_size: int = DEFAULT_BATCH_SIZE, threads: Any = 1, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        import pyroid.filters as pf

        self._pyroid = pf.PIIReplacer(
            remove_emails=self.remove_emails,
            remove_ips=self.remove_ips,
            email_replacement=tuple(self.emails_replacer.replacements),
            ip_replacement=tuple(self.ip_replacer.replacements),
        )
        self.batch_size = _batch_size(batch_size)
        self.threads = threads

    def _format_batch(self, texts: list[str]) -> list[str]:
        native = cast(Any, self._pyroid)._native
        out, self.emails_replacer._replace_i, self.ip_replacer._replace_i = native.run(
            texts,
            self.emails_replacer._replace_i,
            self.ip_replacer._replace_i,
            self.ip_replacer.validator,
            threads=self.threads,
        )
        return cast(list[str], out)

    def format(self, text: str) -> str:
        """One document (datatrove's API); :meth:`run` batches instead."""
        return self._format_batch([text])[0]

    def run(self, data: Iterable[Any], rank: int = 0, world_size: int = 1) -> Any:
        """Rewrite ``doc.text`` of every document, ``batch_size`` at a time."""
        for batch in batched(data, self.batch_size):
            with self.track_time("batch"):
                texts = self._format_batch([d.text for d in batch])
            for doc, text in zip(batch, texts, strict=True):
                self.stat_update(StatHints.total)
                doc.text = text
                yield doc


PIIFormatter.__init__.__signature__ = _signature_of(_dt_pii.PIIFormatter, *_EXTRA)  # type: ignore[attr-defined]
# ---- end PIIFormatter ----

# ---- C4BadWordsFilter ----
__all__ += ["C4BadWordsFilter"]


class C4BadWordsFilter(_Step, _dt_c4.C4BadWordsFilter):
    """datatrove's ``C4BadWordsFilter`` (same arguments) on ``pyroid.filters``.

    The word search runs in Rust, ``batch_size`` documents at a time; the
    lists are loaded exactly as datatrove loads them (its download cache and
    pinned URLs). Everything else is datatrove's own: ``self.uniform`` (its
    ``default_rng(seed)``) is drawn once per flagged document in order, and the
    same statistics are counted. A batch that contains a document whose
    language has no list raises before any of the batch is yielded (datatrove
    yields the documents before it first).
    """

    def __init__(
        self, *args: Any, batch_size: int = DEFAULT_BATCH_SIZE, threads: Any = 1, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = _batch_size(batch_size)
        self.threads = threads
        self._pyroid_lists: dict[str, int] = {}
        self._pyroid_native: list[Any] = []

    def _pyroid_list(self, lang: str) -> int:
        """Datatrove's ``_get_badwords`` without the regex: an index, or -1 (keep)."""
        if lang in self._pyroid_lists:
            return self._pyroid_lists[lang]
        if lang not in _dt_c4._BADWORDS_LANGS:
            if self.fail_on_missing_language:
                raise ValueError(
                    f'There is not badwords list available for "{lang}". '
                    f"Set fail_on_missing_language=False to continue anyway."
                )
            self._pyroid_lists[lang] = -1
            return -1
        local_path = _dt_c4.cached_asset_path_or_download(
            _dt_c4._BADWORDS_URL + lang if lang != "en" else _dt_c4._EN_BADWORDS_URL,
            namespace="filters",
            subfolder="c4_badwords",
        )
        badwords: set[str] = set()
        with open(local_path) as f:  # as datatrove: text mode, locale encoding
            badwords.update(line.strip() for line in f)
        for allowlist in _dt_c4._BADWORDS_ALLOWLIST.values():
            badwords -= allowlist
        from pyroid._pyroid import filters as _native

        self._pyroid_native.append(
            _native._BadWords(list(badwords), lang not in ("ja", "th", "zh"))
        )
        self._pyroid_lists[lang] = len(self._pyroid_native) - 1
        return self._pyroid_lists[lang]

    def filter(self, doc: Any) -> Any:
        """One document (datatrove's API); :meth:`filter_batch` batches instead."""
        return self.filter_batch([doc])[0]

    def filter_batch(self, batch: list[Any]) -> list[Any]:
        """One pyroid search for the batch, then datatrove's decision per document."""
        import pyroid.filters as pf
        from pyroid._pyroid import filters as _native

        langs = [doc.metadata.get("language", self.default_language) for doc in batch]
        ids = [self._pyroid_list(lang) for lang in langs]
        found, _ = _native._badwords_run(
            [d.text for d in batch],
            self._pyroid_native,
            -1,
            ids,
            0.0,
            None,
            "found",
            cast(Any, pf)._unicode_version(),
            threads=self.threads,
        )
        out: list[Any] = []
        for lang, i, hit in zip(langs, ids, found, strict=True):
            if i < 0:
                self.stat_update("missing_badwords_lang", f"missing_badwords_lang_{lang}")
                out.append(True)
            elif hit:
                self.stat_update("documents_with_badwords", f"documents_with_badwords_{lang}")
                if self.keep_fraction > 0.0 and self.uniform() < self.keep_fraction:
                    self.stat_update(
                        "document_kept_with_badwords", f"document_kept_with_badwords_{lang}"
                    )
                    out.append(True)
                else:
                    self.stat_update(f"document_removed_with_badwords_{lang}")
                    out.append((False, "document_removed_with_badwords"))
            else:
                out.append(True)
        return out


C4BadWordsFilter.__init__.__signature__ = _signature_of(_dt_c4.C4BadWordsFilter, *_EXTRA)  # type: ignore[attr-defined]
# ---- end C4BadWordsFilter ----

# ---- URLFilter ----
# ---- end URLFilter ----

# ---- SymbolLinesFormatter ----
from datatrove.pipeline.formatters import symbol_lines_remover as _dt_symbols  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: E402

__all__ += ["SymbolLinesFormatter"]


class SymbolLinesFormatter(_Step, _dt_symbols.SymbolLinesFormatter):
    r"""datatrove's ``SymbolLinesFormatter`` (same arguments) on ``pyroid.filters``.

    ``doc.text`` becomes the same text, byte for byte, computed ``batch_size``
    documents at a time (FineWeb-2 runs it as
    ``SymbolLinesFormatter(symbols_to_remove=["|"], replace_char="\n")``).
    """

    def __init__(
        self, *args: Any, batch_size: int = DEFAULT_BATCH_SIZE, threads: Any = 1, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = _batch_size(batch_size)
        self.threads = threads

    def _native(self) -> Any:
        import pyroid.filters as pf

        cfg = pf.SymbolLines(list(self.symbols_to_remove), self.replace_char)
        return cast(Any, pf)._symbol_lines_native(cfg)

    def format(self, text: str) -> str:
        """One document (datatrove's API); :meth:`run` batches instead."""
        return cast(str, self._native().run(text))

    def run(self, data: Iterable[Any], rank: int = 0, world_size: int = 1) -> Any:
        """Rewrite ``doc.text`` of every document, ``batch_size`` at a time."""
        native = self._native()
        for batch in batched(data, self.batch_size):
            with self.track_time("batch"):
                texts = native.run([d.text for d in batch], threads=self.threads)
            for doc, text in zip(batch, texts, strict=True):
                self.stat_update(StatHints.total)
                doc.text = text
                yield doc


SymbolLinesFormatter.__init__.__signature__ = _signature_of(  # type: ignore[attr-defined]
    _dt_symbols.SymbolLinesFormatter, *_EXTRA
)
# ---- end SymbolLinesFormatter ----

# ---- UnigramLogProbFilter ----
# ---- end UnigramLogProbFilter ----

# ---- SentenceDedup ----
# ---- end SentenceDedup ----
