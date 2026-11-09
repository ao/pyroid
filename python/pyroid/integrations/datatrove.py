# Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.
# mypy: disallow-subclassing-any=False
# (datatrove ships no type information: its classes are Any to mypy.)
"""Drop-in datatrove pipeline steps backed by pyroid.

Change the import and leave the pipeline as it is::

    # from datatrove.pipeline.filters import GopherQualityFilter, LanguageFilter
    from pyroid.integrations.datatrove import GopherQualityFilter, LanguageFilter

Each class here subclasses datatrove's own step of the same name and takes the
same constructor arguments, plus two keyword-only ones:

- ``batch_size``: documents per pyroid call (default 1000).
- ``threads``: pyroid's ``threads=`` (default 1, since datatrove already runs
  one task per process). Pass ``None`` for pyroid's whole thread pool when you
  run few tasks.

The documents that come out are the same as with datatrove's step: same
``text``, same ``metadata``, same keep/drop decisions and drop reasons, so
exclusion writers and later steps see the same data. The work happens in Rust,
in batches, without the GIL:

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
- :class:`PIIFormatter`, :class:`SymbolLinesFormatter`: ``pyroid.filters``
  (same text, same replacement rotation).
- :class:`C4BadWordsFilter`, :class:`URLFilter`: ``pyroid.filters`` (same
  decisions and reasons; datatrove's own random draws, word lists and
  blocklists).
- :class:`SentenceDedupSignature`, :class:`SentenceFindDedups`,
  :class:`SentenceDedupFilter`, :class:`SentenceDedupBuildIndex`:
  ``pyroid.dedup``; the same signature, duplicate and index files, byte for
  byte, and the same documents out.
- :class:`Trafilatura`: ``pyroid.trafilatura`` (byte-identical to the installed trafilatura,
  1.11.0 as datatrove installs it, or 2.3.1; in this process, with no timeout).

Languages: the filter and MinHash steps take datatrove's ``language`` (a
datatrove language code such as ``"fra_Latn"``). Every code pyroid tokenizes
natively, which is 4,013 of datatrove's 4,343 with a SudachiDict dictionary
installed and 4,009 without (``pyroid.filters.languages()``, ``Language.native``),
runs in Rust with that language's datatrove tokenizer reproduced exactly. Any
other code (Vietnamese, Thai, the Stanza languages, Japanese without a
dictionary, ...), a code datatrove has no tokenizer for, or a
``WordTokenizer`` object runs datatrove's own code for that step, with a
one-time :class:`pyroid.FallbackWarning`.

What differs: per-line statistics of ``C4QualityFilter`` (``line-total``,
``line-filter-*``) are not counted, and batched steps also count ``batches``.

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

        if not _tokenizes(self.language):
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


_NATIVE: list[frozenset[str]] = []


def _tokenizes(language: object) -> bool:
    """Does pyroid tokenize ``language`` (a datatrove code, or a tokenizer object) natively?

    True for the codes ``pyroid.filters.languages()`` marks ``native``; False for the others,
    for codes datatrove has no tokenizer for, and for ``WordTokenizer`` objects.
    """
    if not _NATIVE:
        import pyroid.filters as pf

        _NATIVE.append(frozenset(code for code, lang in pf.languages().items() if lang.native))
    return isinstance(language, str) and language in _NATIVE[0]


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
    them unchanged. spaCy is not needed. ``language`` is datatrove's: shingles
    come from that language's word tokenizer, in Rust for every code pyroid
    tokenizes natively; another one runs datatrove's own step
    (:class:`pyroid.FallbackWarning`). Note that datatrove's sort pass logs
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
            if not _tokenizes(bound.arguments["language"]) or not hasattr(self, "_hash_func"):
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
                    [d.text for d in batch], cfg, language=self.language, threads=self.threads
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
        if not _tokenizes(self.language):
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
        self._pyroid: Any = None
        self._replacer()  # check the replacements now, as datatrove's own __init__ does
        self.batch_size = _batch_size(batch_size)
        self.threads = threads

    def _replacer(self) -> Any:
        """The native replacer, built on first use (and again after unpickling)."""
        if self._pyroid is None:
            import pyroid.filters as pf

            self._pyroid = pf.PIIReplacer(
                remove_emails=self.remove_emails,
                remove_ips=self.remove_ips,
                email_replacement=tuple(self.emails_replacer.replacements),
                ip_replacement=tuple(self.ip_replacer.replacements),
            )
        return self._pyroid

    def __getstate__(self) -> dict[str, Any]:
        # The native replacer does not pickle (datatrove deep-copies or pickles steps to its
        # workers): it is rebuilt from the same settings; the rotation counters travel along.
        state = self.__dict__.copy()
        state["_pyroid"] = None
        return state

    def _format_batch(self, texts: list[str]) -> list[str]:
        native = cast(Any, self._replacer())._native
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

    def __getstate__(self) -> dict[str, Any]:
        # The native word lists do not pickle: they are reloaded in the worker, as datatrove
        # reloads its own.
        state = self.__dict__.copy()
        state["_pyroid_lists"] = {}
        state["_pyroid_native"] = []
        return state

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
from datatrove.pipeline.filters import url_filter as _dt_url  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: E402

__all__ += ["URLFilter"]


class URLFilter(_Step, _dt_url.URLFilter):
    """datatrove's ``URLFilter`` (same arguments) on :class:`pyroid.filters.URLFilter`.

    The same keep/drop decisions and ``filter_reason`` strings (``domain``,
    ``subdomain``, ``url``, ``hard_blacklisted``, ``soft_blacklisted``,
    ``blacklisted_subword``), from ``doc.metadata["url"]``, ``batch_size`` URLs per
    pyroid call. The suffix lists are this step's own ``tldextractor``'s (its
    cached Public Suffix List or tldextract's snapshot), and with
    ``use_integrated_lists`` the block lists are datatrove's, extracted into
    datatrove's cache as datatrove does; both are loaded at the first batch, in the
    worker. pyahocorasick is not needed.

    What differs: a document without a ``url`` raises datatrove's
    ``AssertionError`` when its batch is filtered (before any document of that batch
    is passed on); datatrove's per-document ``DeprecationWarning`` from tldextract's
    ``registered_domain`` (hidden by Python's default filters) is not emitted;
    ``block_listed_domains`` and the other list attributes hold the constructor's
    entries only (the integrated lists live in Rust); ``banned_subwords_automaton``
    is ``None``.
    """

    _pyroid_requires = ("tldextract", "fasteners")

    def __init__(
        self, *args: Any, batch_size: int = DEFAULT_BATCH_SIZE, threads: Any = 1, **kwargs: Any
    ) -> None:
        # datatrove's __init__, without its pyahocorasick automaton.
        bound = inspect.signature(_dt_url.URLFilter.__init__).bind(self, *args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        _dt_url.BaseFilter.__init__(self, a["exclusion_writer"])
        from tldextract import TLDExtract  # type: ignore[import-not-found,unused-ignore]

        parse = _dt_url.parse_list
        self.soft_word_threshold = a["soft_word_threshold"]
        extra_domains, extra_urls = a["extra_domains"], a["extra_urls"]
        self.block_listed_domains = (
            parse(extra_domains, do_normalize=False) if extra_domains else set()
        )
        self.block_listed_url = parse(extra_urls, do_normalize=False) if extra_urls else set()
        self.banned_words = parse(a["banned_words"]) if a["banned_words"] else set()
        self.banned_subwords = parse(a["banned_subwords"]) if a["banned_subwords"] else set()
        self.soft_banned_words = parse(a["soft_banned_words"]) if a["soft_banned_words"] else set()
        self.use_integrated_lists = a["use_integrated_lists"]
        self._downloaded = False
        self.tldextractor = TLDExtract()
        self.banned_subwords_automaton = None
        self.batch_size = _batch_size(batch_size)
        self.threads = threads
        self._pyroid: Any = None

    def download_data(self) -> None:
        """Build the pyroid filter: the lists, and datatrove's if ``use_integrated_lists``."""
        if self._pyroid is not None:
            return
        import pyroid.filters as pf

        self._pyroid = cast(Any, pf.URLFilter)._from_parsed(
            self.soft_word_threshold,
            self.block_listed_domains,
            self.block_listed_url,
            self.banned_words,
            self.banned_subwords,
            self.soft_banned_words,
            self.use_integrated_lists,
            self.tldextractor,
        )
        self._downloaded = bool(self.use_integrated_lists)

    def filter(self, doc: Any) -> Any:
        """One document (datatrove's API); :meth:`filter_batch` batches instead."""
        return self.filter_batch([doc])[0]

    def filter_batch(self, batch: list[Any]) -> list[Any]:
        """One pyroid call for the batch's URLs."""
        self.download_data()
        urls = []
        for doc in batch:
            url = doc.metadata.get("url")
            assert url, "Document does not have url in its metadata"
            urls.append(url)
        reasons = self._pyroid._reasons_any(urls, self.threads)
        return [True if r is None else (False, r) for r in reasons]

    def __getstate__(self) -> dict[str, Any]:
        # The native filter is rebuilt in each worker, as datatrove reloads its lists.
        state = self.__dict__.copy()
        state["_pyroid"] = None
        state["_downloaded"] = False
        return state


URLFilter.__init__.__signature__ = _signature_of(_dt_url.URLFilter, *_EXTRA)  # type: ignore[attr-defined]
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

# ---- SentenceDedup ----
# datatrove's sentence dedup: stage 1 and 3 on pyroid.dedup, stage 2 (and the index builder) a
# merge in Rust. Same files, byte for byte; same documents, texts and exclusion-writer documents.

import contextlib  # noqa: E402
import dataclasses  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

from datatrove.pipeline.dedup import sentence_dedup as _dt_sd  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: E402
from datatrove.utils.typeshelper import ExtensionHelperSD as _ExtSD  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: E402

__all__ += [
    "SentenceDedupBuildIndex",
    "SentenceDedupFilter",
    "SentenceDedupSignature",
    "SentenceFindDedups",
]


def _sd_config(step: str, config: Any) -> Any:
    """The pyroid ``SentDedupConfig`` for datatrove's, or ``None`` (fallback warned)."""
    import pyroid.dedup as pd

    try:
        return pd._sent_config(config, {})
    except (TypeError, ValueError) as e:
        _fallback(step, f"this SentDedupConfig ({e})")
        return None


def _sd_native(step: str, config: Any, language: Any, stage3: bool) -> Any:
    """The native config for a stage, or ``None`` to run datatrove's code (fallback warned).

    ``language`` is a datatrove language code or (as datatrove allows) a ``WordTokenizer``;
    pyroid tokenizes the codes ``pyroid.filters.languages()`` marks native (lines need no
    tokenizer); for the others, and for a custom tokenizer, datatrove's own code runs.
    """
    import pyroid.dedup as pd

    cfg = _sd_config(step, config)
    if cfg is None:
        return None
    native = pd._native_sent(cfg, language if isinstance(language, str) else "")
    if native.needs_tokenizer(stage3) and not native.has_tokenizer:
        if isinstance(language, str):
            _fallback(step, f"language={language!r}")
        else:
            _fallback(step, "a custom word tokenizer")
        return None
    return native


def _sd_source(folder: Any, path: str) -> Any:
    """A file for the native merge: its local path (read in Rust), else its bytes."""
    if folder.is_local():
        return folder.resolve_paths(path)
    with folder.open(path, "rb") as f:
        return f.read()


class _SentenceStep(_Step):
    batch_size: int
    threads: Any

    def _setup(self, batch_size: int, threads: Any) -> None:
        self.batch_size = _batch_size(batch_size)
        self.threads = threads


class SentenceDedupSignature(_SentenceStep, _dt_sd.SentenceDedupSignature):
    """datatrove's ``SentenceDedupSignature`` (stage 1) on ``pyroid.dedup``.

    Writes the same ``{bucket:04d}/{rank:05d}.c4_sig`` files, byte for byte (the
    same buckets, the same empty files), so datatrove's own later stages run on them
    unchanged. spaCy is not needed. ``language`` is datatrove's: every code pyroid tokenizes
    natively (``pyroid.filters.languages()``) runs in Rust; another one with
    ``split_sentences=True`` runs datatrove's own code (:class:`pyroid.FallbackWarning`).
    """

    def __init__(
        self, *args: Any, batch_size: int = DEFAULT_BATCH_SIZE, threads: Any = 1, **kwargs: Any
    ) -> None:
        try:
            super().__init__(*args, **kwargs)
        except ImportError:
            # datatrove loads its spaCy word tokenizer last; pyroid has its own.
            if not _tokenizes(getattr(self, "language", None)):
                raise
            self.tokenizer = None
        self._setup(batch_size, threads)

    def run(self, data: Iterable[Any], rank: int = 0, world_size: int = 1) -> None:
        """Signatures of this rank's documents, ``batch_size`` at a time; then the files."""
        native = _sd_native("SentenceDedupSignature", self.config, self.language, stage3=False)
        if native is None:
            return super().run(data, rank, world_size)  # type: ignore[no-any-return]
        import pyroid.dedup as pd

        state = pd._native._SentenceState(native)
        overflow: OverflowError | None = None
        for batch in batched(data, self.batch_size):
            for _ in batch:
                self.stat_update(StatHints.total)
            with self.track_time("batch"):
                try:
                    state.add([d.text for d in batch], threads=self.threads)
                except OverflowError as e:
                    # A window index past numpy's <u2: datatrove only fails when it saves
                    # the signatures, once every document has gone through the pipeline.
                    overflow = overflow or e
        if overflow is not None:
            raise overflow
        for bucket, blob in enumerate(state.signature_files(self.finder_workers)):
            name = f"{bucket:04d}/{rank:05d}{_ExtSD.stage_1_signature}"
            with self.output_folder.open(name, mode="wb") as f:
                f.write(blob)
        return None


class SentenceFindDedups(_SentenceStep, _dt_sd.SentenceFindDedups):
    """datatrove's ``SentenceFindDedups`` (stage 2) as a merge in Rust.

    Reads the same signature (and index) files and writes the same
    ``{rank:04d}/{stem}.c4_dup`` files, byte for byte; local files are read by
    Rust directly, remote ones through fsspec. ``batch_size`` and ``threads`` are
    accepted for uniformity (the merge is one sequential pass).
    """

    def __init__(
        self, *args: Any, batch_size: int = DEFAULT_BATCH_SIZE, threads: Any = 1, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._setup(batch_size, threads)

    def run(self, data: Any = None, rank: int = 0, world_size: int = 1) -> None:
        """Find the duplicate windows of this finder task's hash bucket."""
        cfg = _sd_config("SentenceFindDedups", self.config)
        if cfg is None:
            return super().run(data, rank, world_size)  # type: ignore[no-any-return]
        import pyroid.dedup as pd

        ext = _ExtSD.stage_1_signature
        with self.stats.time_stats:
            if world_size == 1:
                sig_files = self.data_folder.list_files(glob_pattern="*/*" + ext)
                if any(not sig_file.startswith("0000/") for sig_file in sig_files):
                    raise ValueError(
                        f"{world_size=} but found sig files for different hash buckets. "
                        "Set tasks=finder_workers"
                    )
            else:
                sig_files = self.data_folder.list_files(
                    subdirectory=f"{rank:04d}", glob_pattern=ext
                )
            index_files = self.index_folder.list_files() if self.index_folder else None
            out = pd._native._sentence_find_dups(
                [_sd_source(self.data_folder, f) for f in sig_files],
                [_sd_source(self.index_folder, f) for f in index_files or ()],
                precision=cfg.hash_config.precision,
                only_dedup_in_index=cfg.only_dedup_in_index,
            )
            for sig_file, blob in zip(sig_files, out, strict=True):
                if blob:
                    stem = sig_file.rsplit("/", 1)[-1].removesuffix(ext)
                    name = f"{rank:04d}/{stem}{_ExtSD.stage_2_duplicates}"
                    with self.output_folder.open(name, mode="wb") as f:
                        f.write(blob)
        return None


class SentenceDedupFilter(_SentenceStep, _dt_sd.SentenceDedupFilter):
    """datatrove's ``SentenceDedupFilter`` (stage 3) on ``pyroid.dedup``.

    Reads the same ``.c4_dup`` files and yields the same documents with the same
    texts; the exclusion writer gets the same documents (``>>>`` / ``<<<``-marked
    texts). Only documents with duplicates go to Rust, ``batch_size`` at a time.
    spaCy is not needed. ``language`` is datatrove's: a code pyroid does not tokenize
    natively (``pyroid.filters.languages()``) that needs the tokenizer runs datatrove's own
    code (:class:`pyroid.FallbackWarning`).
    """

    def __init__(
        self, *args: Any, batch_size: int = DEFAULT_BATCH_SIZE, threads: Any = 1, **kwargs: Any
    ) -> None:
        try:
            super().__init__(*args, **kwargs)
        except ImportError:
            # datatrove loads its spaCy word tokenizer before the last two attributes.
            bound = inspect.signature(_dt_sd.SentenceDedupFilter.__init__).bind(
                self, *args, **kwargs
            )
            bound.apply_defaults()
            if not _tokenizes(bound.arguments["language"]) or not hasattr(self, "config"):
                raise
            self.tokenizer = None
            self.exclusion_writer = bound.arguments["exclusion_writer"]
            self.language = bound.arguments["language"]
        self._setup(batch_size, threads)

    def _read_dups(self, rank: int) -> Any:
        import pyroid.dedup as pd

        folders = self.data_folder.list_files(include_directories=True, recursive=False)
        ext = _ExtSD.stage_2_duplicates
        files = [
            f
            for f in [f"{folder}/{rank:05d}{ext}" for folder in folders]
            if self.data_folder.exists(f)
        ]

        def read(path: str) -> bytes:
            with self.data_folder.open(path, "rb") as f:
                data: bytes = f.read()
            return data

        with ThreadPoolExecutor() as pool:
            blobs = list(pool.map(read, files))
        return pd._native._SentenceDups.parse(blobs)

    def run(self, data: Iterable[Any], rank: int = 0, world_size: int = 1) -> Any:
        """Remove this rank's duplicate windows, ``batch_size`` documents at a time."""
        native = _sd_native("SentenceDedupFilter", self.config, self.language, stage3=True)
        if native is None:
            yield from super().run(data, rank, world_size)
            return
        dups = self._read_dups(rank)
        writer_cm = self.exclusion_writer if self.exclusion_writer else contextlib.nullcontext()
        with writer_cm as writer:
            start = 0
            for batch in batched(data, self.batch_size):
                ids = dups.docs_in(start, start + len(batch))
                done: dict[int, Any] = {}
                if ids:
                    with self.track_time("batch"):
                        text, keep, formatted, stats = dups.filter(
                            [batch[i - start].text for i in ids],
                            native,
                            ids=ids,
                            details=True,
                            threads=self.threads,
                        )
                    done = dict(
                        zip(ids, zip(text, keep, formatted, stats, strict=True), strict=True)
                    )
                for i, doc in enumerate(batch):
                    self.stat_update(StatHints.total)
                    r = done.get(start + i)
                    if r is None:
                        filtered, kept, original_formatted = doc.text, bool(doc.text), None
                    else:
                        filtered, kept, original_formatted, (spans, n_kept) = r
                        if n_kept < spans:
                            self.stat_update("removed_sentences", value=spans - n_kept)
                        self.stat_update("original_sentences", value=spans)
                    if kept:
                        self.update_doc_stats(doc)
                        if filtered != doc.text and writer:
                            writer.write(
                                dataclasses.replace(doc, text=original_formatted), rank=rank
                            )
                        doc.text = filtered
                        yield doc
                    elif writer:
                        doc.text = original_formatted
                        writer.write(doc, rank=rank)
                start += len(batch)


class SentenceDedupBuildIndex(_SentenceStep, _dt_sd.SentenceDedupBuildIndex):
    """datatrove's ``SentenceDedupBuildIndex`` as a merge in Rust (the same index file).

    Like datatrove's own step, it fails (``StopIteration``) on an empty signature file:
    that case runs datatrove's code. ``batch_size`` and ``threads`` are accepted for
    uniformity.
    """

    def __init__(
        self, *args: Any, batch_size: int = DEFAULT_BATCH_SIZE, threads: Any = 1, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._setup(batch_size, threads)

    def run(self, data: Any = None, rank: int = 0, world_size: int = 1) -> None:
        """Write the distinct hashes of every signature file."""
        assert world_size == 1, "SentenceDedupBuildIndex can only run on a single worker."
        cfg = _sd_config("SentenceDedupBuildIndex", self.config)
        if cfg is None:
            return super().run(data, rank, world_size)  # type: ignore[no-any-return]
        import pyroid.dedup as pd

        sig_files = self.data_folder.list_files(glob_pattern="*/*" + _ExtSD.stage_1_signature)
        if any(self.data_folder.size(f) == 0 for f in sig_files):
            # datatrove's `[next(sig_reader) for sig_reader in sig_readers]`: StopIteration.
            return super().run(data, rank, world_size)  # type: ignore[no-any-return]
        with self.stats.time_stats:
            blob = pd._native._sentence_build_index(
                [_sd_source(self.data_folder, f) for f in sig_files],
                precision=cfg.hash_config.precision,
            )
            with self.output_folder.open(f"{self.index_name}.{_ExtSD.index}", mode="wb") as f:
                f.write(blob)
        return None


for _sd_cls, _sd_parent in (
    (SentenceDedupSignature, _dt_sd.SentenceDedupSignature),
    (SentenceFindDedups, _dt_sd.SentenceFindDedups),
    (SentenceDedupFilter, _dt_sd.SentenceDedupFilter),
    (SentenceDedupBuildIndex, _dt_sd.SentenceDedupBuildIndex),
):
    cast(Any, _sd_cls.__init__).__signature__ = _signature_of(_sd_parent, *_EXTRA)
del _sd_cls, _sd_parent
# ---- end SentenceDedup ----

# ---- Trafilatura ----
from datatrove.pipeline.extractors import trafilatura as _dt_traf  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: E402
from datatrove.utils.logging import logger as _dt_logger  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: E402

__all__ += ["Trafilatura"]

#: Default ``batch_size`` of :class:`Trafilatura`: HTML pages are large, so batches are smaller.
TRAFILATURA_BATCH_SIZE = 128
_TIMEOUT = object()  # a page datatrove's step would count as "timeout"


def _installed_trafilatura() -> str | None:
    """The installed trafilatura's version (not imported), or None."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("trafilatura")
    except PackageNotFoundError:
        return None


class Trafilatura(_Step, _dt_traf.Trafilatura):
    """datatrove's ``Trafilatura`` extractor (same arguments and defaults) on pyroid.trafilatura.

    ``doc.text`` becomes ``trafilatura.extract(doc.text, favor_precision=favour_precision,
    include_comments=False, deduplicate=deduplicate, **kwargs)``, byte for byte, computed in
    this process ``batch_size`` documents at a time; the same documents are forwarded and
    dropped, with the same statistics (``total``, ``extracted``, ``forwarded``, ``dropped``,
    ``doc_len``). Errors count as datatrove counts them: a page whose extraction raises is
    ``broken_process`` (it kills datatrove's child process, and the next child starts with an
    empty dedup cache, so the rest of the task does here too), and arguments that make
    ``trafilatura.extract("")`` raise (datatrove's warm-up call) make every page ``timeout``.
    Trafilatura itself is not needed.

    Extra keyword-only arguments: ``compat``, the trafilatura release to reproduce (``None``:
    the installed trafilatura's version when pyroid reproduces it, which includes the 1.11.0 that
    ``datatrove[processing]`` installs, or pyroid's default, 2.3.1, when trafilatura is not
    installed; for any other installed version the documents go to datatrove's own step, with a
    one-time :class:`pyroid.FallbackWarning`), ``batch_size`` and ``threads``.

    What differs: there is no child process and no ``timeout`` (pages datatrove gives up on
    after ``timeout`` seconds, counting ``timeout`` and restarting its dedup cache, get their
    text here). With ``deduplicate=True`` the dedup cache is a fresh
    :class:`pyroid.trafilatura.LRUCache` per :meth:`run`, i.e. per task, as datatrove's per-task
    child process has a fresh one. Time is tracked per batch.
    """

    def __init__(
        self,
        favour_precision: bool = True,
        include_images: bool = False,
        timeout: float = 1,
        deduplicate: Any = True,
        *,
        compat: str | None = None,
        batch_size: int = TRAFILATURA_BATCH_SIZE,
        threads: Any = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(favour_precision, include_images, timeout, deduplicate, **kwargs)
        import pyroid.trafilatura as pt

        if compat is not None:
            cast(Any, pt)._mode(compat)  # ValueError for a release pyroid does not reproduce
        self.compat = compat
        self.batch_size = _batch_size(batch_size)
        self.threads = threads
        self._datatrove = False  # running datatrove's own step (see run)

    def _mode(self) -> str | None:
        """The release to reproduce, or None to run datatrove's own step."""
        import pyroid.trafilatura as pt

        if self.compat is not None:
            return self.compat
        installed = _installed_trafilatura()
        if installed is None:
            return pt.TRAFILATURA_VERSION
        return installed if installed in pt.TRAFILATURA_VERSIONS else None

    def _call_kwargs(self, deduplicate: Any) -> dict[str, Any]:
        """The keyword arguments datatrove passes to ``trafilatura.extract``."""

        def extract(**kw: Any) -> dict[str, Any]:  # named for datatrove's TypeError messages
            return kw

        return extract(
            favor_precision=self.favour_precision,
            include_comments=False,
            deduplicate=deduplicate,
            **self.kwargs,
        )

    def extract(self, text: str) -> str:
        """One document (datatrove's API); :meth:`run` batches instead."""
        if self._datatrove:
            return cast(str, _dt_traf.Trafilatura.extract(self, text))
        import pyroid.trafilatura as pt

        mode = self._mode() or pt.TRAFILATURA_VERSION
        many = cast(Any, pt)._extract_many
        kwargs = self._call_kwargs(self.deduplicate)
        return cast(str, many([text], kwargs, 1, mode, raise_errors=True)[0])

    def _task_cache(self, mode: str) -> Any:
        """The dedup cache of one task, or None when ``deduplicate`` names its own cache.

        It stands for trafilatura's module-level cache, which datatrove's per-task child process
        has fresh. trafilatura 2.3.1 also takes a cache as ``deduplicate``: that one is used.
        """
        import pyroid.trafilatura as pt

        dedup = self.deduplicate
        if not dedup:
            return None
        if mode == pt.TRAFILATURA_VERSION and (
            isinstance(dedup, pt.LRUCache) or cast(Any, pt)._is_real_lru(dedup)
        ):
            return None
        return pt.LRUCache(maxsize=pt.LRU_SIZE)

    def _batch(
        self, texts: list[Any], kwargs: dict[str, Any], mode: str, cache: Any
    ) -> tuple[list[Any], Any]:
        """The results for `texts` (exceptions in place) and the task cache for the next batch.

        A page whose extraction raises kills datatrove's child process, and the next child starts
        with an empty dedup cache: so does the rest of the task here, when the cache is the task's.
        """
        import pyroid.trafilatura as pt

        many = cast(Any, pt)._extract_many
        out: list[Any] = []
        while True:
            results = many(texts, kwargs, self.threads, mode, raise_errors=False, task_cache=cache)
            bad = next((i for i, r in enumerate(results) if isinstance(r, Exception)), None)
            if bad is None or cache is None:
                return out + results, cache
            out += results[: bad + 1]
            cache = pt.LRUCache(maxsize=pt.LRU_SIZE)
            texts = texts[bad + 1 :]
            if not texts:
                return out, cache

    def run(self, data: Iterable[Any], rank: int = 0, world_size: int = 1) -> Any:
        """Extract the text of every document, ``batch_size`` at a time."""
        import pyroid.trafilatura as pt

        mode = self._mode()
        if mode is None:
            _fallback(
                "Trafilatura",
                f"trafilatura {_installed_trafilatura()} (pyroid reproduces "
                f"{', '.join(pt.TRAFILATURA_VERSIONS)})",
            )
            self._datatrove = True
            try:
                yield from super().run(data, rank, world_size)
            finally:
                self._datatrove = False
            return
        kwargs: dict[str, Any] | None = None
        cache = self._task_cache(mode)
        warmed = False
        for batch in batched(data, self.batch_size):
            with self.track_time("batch"):
                if not warmed:
                    # datatrove's child calls extract("") before its first page; if that raises,
                    # no child ever starts and every page times out
                    warmed = True
                    try:
                        kwargs = self._call_kwargs(self.deduplicate)
                        cast(Any, pt)._extract_many([""], kwargs, 1, mode, raise_errors=True)
                    except Exception:
                        kwargs = None
                if kwargs is None:
                    results: list[Any] = [_TIMEOUT] * len(batch)
                else:
                    texts = [d.text for d in batch]
                    results, cache = self._batch(texts, kwargs, mode, cache)
            for doc, text in zip(batch, results, strict=True):
                self.stat_update(StatHints.total)
                if text is _TIMEOUT:
                    self.stat_update("timeout")
                    _dt_logger.warning("⏰ Timeout while cleaning record text. Skipping record.")
                    continue
                if isinstance(text, Exception):
                    self.stat_update("broken_process")
                    _dt_logger.warning(
                        "Process died unexpectedly, will create new process for next document"
                    )
                    continue
                self.stat_update("extracted")
                doc.text = text
                if doc.text:
                    self.stat_update(StatHints.forwarded)
                    self.update_doc_stats(doc)
                    yield doc
                else:
                    self.stat_update(StatHints.dropped)


Trafilatura.__init__.__signature__ = _signature_of(  # type: ignore[attr-defined]
    _dt_traf.Trafilatura,
    inspect.Parameter("compat", inspect.Parameter.KEYWORD_ONLY, default=None),
    inspect.Parameter("batch_size", inspect.Parameter.KEYWORD_ONLY, default=TRAFILATURA_BATCH_SIZE),
    inspect.Parameter("threads", inspect.Parameter.KEYWORD_ONLY, default=1),
)
# ---- end Trafilatura ----
