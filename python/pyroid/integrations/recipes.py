"""Cleaning recipes: the FineWeb chain, natively in pyroid.

:class:`FineWeb` is the chain the pyroid benchmark gates against datatrove
(``bench/chain.py``), over Arrow batches, as a reusable object::

    ftfy.fix_text (ftfy's default config)
    -> LanguageFilter(languages=["en"], language_threshold=0.65), fastText lid.176.bin
    -> GopherRepetitionFilter() -> GopherQualityFilter()
    -> C4QualityFilter(filter_no_terminal_punct=False) -> FineWebQualityFilter()
    -> MinHash dedup (sha1/64, 14 buckets x 8 hashes, word 5-grams) over the survivors, in order

Per batch, :meth:`FineWeb.annotate` runs every stage but the dedup and returns
Arrow columns (no Python ``str`` is built for the texts). Dedup is a property
of the whole corpus: feed the survivors of each batch, in order, to the index
from :meth:`FineWeb.new_dedup` and read its keep mask at the end. This is what
``pyroid clean --recipe fineweb`` does.

:class:`FineWebWarc` is datatrove's ``examples/fineweb.py`` from WARC files (``pyroid clean
--recipe fineweb-warc``): :mod:`pyroid.warc` reads the pages, then the URL filter,
:class:`TaskExtractor` (datatrove's Trafilatura step), :class:`FineWeb` without ftfy, MinHash and
PII replacement.

Needs pyarrow and numpy (``pip install 'pyroid[cli]'``).
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
import shutil
import tempfile
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import pyarrow as pa
import pyarrow.compute as pc

import pyroid.dedup
import pyroid.fasttext
import pyroid.filters as pf
import pyroid.ftfy

if TYPE_CHECKING:
    from pyroid.warc import Document

__all__ = [
    "FINEWEB_TRAFILATURA",
    "LID_176_MIRRORS",
    "LID_176_SHA256",
    "LID_176_URL",
    "MINHASH_CONFIG",
    "Annotation",
    "FineWeb",
    "FineWebWarc",
    "TaskExtractor",
    "WarcAnnotation",
    "lid_model_path",
]

#: Where fastText's language-ID model comes from (CC BY-SA 3.0), and its sha256.
LID_176_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
LID_176_SHA256 = "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"
#: Copies of the same file, tried in order when the download from ``LID_176_URL`` fails (on
#: 2026-11-12 dl.fbaipublicfiles.com served an expired certificate). Each must match
#: ``LID_176_SHA256``.
LID_176_MIRRORS: tuple[str, ...] = (
    "https://huggingface.co/julien-c/fasttext-language-id/resolve/main/lid.176.bin",
)

#: datatrove ``examples/fineweb.py``'s MinHash settings.
MINHASH_CONFIG: dict[str, Any] = {
    "n_grams": 5,
    "num_buckets": 14,
    "hashes_per_bucket": 8,
    "seed": 1,
    "hash_fc": "sha1",
    "precision": 64,
}


def _cache_dir() -> Path:
    if d := os.environ.get("PYROID_CACHE_DIR"):
        return Path(d)
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "pyroid"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def lid_model_path(path: str | os.PathLike[str] | None = None, *, download: bool = True) -> str:
    """The fastText ``lid.176.bin`` file to use.

    Args:
        path: An explicit model file; returned as is.
        download: When neither ``path`` nor ``$PYROID_LID_MODEL`` is set and the
            cached copy is missing, download it from ``LID_176_URL`` (else from
            ``LID_176_MIRRORS``) into the
            cache (``$PYROID_CACHE_DIR``, else ``$XDG_CACHE_HOME/pyroid``, else
            ``~/.cache/pyroid``) and check its sha256.

    Returns:
        The model path.

    Raises:
        FileNotFoundError: No model, and ``download`` is false.
        OSError: The download failed or its sha256 is wrong.
    """
    if path is not None:
        return os.fspath(path)
    if env := os.environ.get("PYROID_LID_MODEL"):
        return env
    target = _cache_dir() / "lid.176.bin"
    if target.exists():
        return str(target)
    if not download:
        raise FileNotFoundError(
            f"no fastText LID model: pass a path, set PYROID_LID_MODEL, or put lid.176.bin in "
            f"{target.parent}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in (LID_176_URL, *LID_176_MIRRORS):
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".lid.176.bin.")
        try:
            try:
                with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url) as resp:
                    shutil.copyfileobj(resp, out, 1 << 20)
            except OSError as e:
                errors.append(f"{url}: {e}")
                continue
            got = _sha256(Path(tmp))
            if got != LID_176_SHA256:
                errors.append(f"{url}: sha256 {got}, expected {LID_176_SHA256}")
                continue
            os.replace(tmp, target)
            return str(target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    raise OSError("could not download lid.176.bin: " + "; ".join(errors))


# datatrove 0.10.1's drop reasons per filter kind ("empty" is shared by two of them).
_REASONS: dict[str, re.Pattern[str]] = {
    "gopher_repetition": re.compile(
        r"empty|dup_(?:para|line)_(?:char_)?frac|top_\d+_gram|duplicated_\d+_n_grams"
    ),
    "gopher_quality": re.compile(r"gopher_\w+"),
    "c4_quality": re.compile(r"lorem_ipsum|curly_bracket|too_few_sentences"),
    "fineweb_quality": re.compile(
        r"empty|line_punct_ratio|short_line_ratio|char_dup_ratio|list_ratio"
    ),
}
_KIND = {
    pf.GopherRepetition: "gopher_repetition",
    pf.GopherQuality: "gopher_quality",
    pf.C4Quality: "c4_quality",
    pf.FineWebQuality: "fineweb_quality",
}


_DATATROVE_KIND = {
    "GopherRepetitionFilter": "gopher_repetition",
    "GopherQualityFilter": "gopher_quality",
    "C4QualityFilter": "c4_quality",
    "FineWebQualityFilter": "fineweb_quality",
}


def _stage_name(cfg: Any) -> str:
    """``gopher_quality`` etc. for a pyroid config or a datatrove filter instance."""
    kind = _KIND.get(type(cfg)) or _DATATROVE_KIND.get(type(cfg).__name__)
    if kind is None:
        raise TypeError(f"not a supported filter: {type(cfg).__name__}")
    return kind


@dataclass(frozen=True)
class Annotation:
    """What :meth:`FineWeb.annotate` returns: one value per input document.

    Attributes:
        text: The text as it leaves the chain (after ftfy and C4's line removal);
            for a dropped document, the text the dropping stage saw.
        keep: True if the document survived every stage but the dedup.
        reason: ``None`` if kept, else ``"lid"`` or ``"<stage>:<datatrove reason>"``
            (e.g. ``"gopher_quality:gopher_short_doc"``); ``"null"`` for a null text.
        language: fastText's top label, without ``__label__`` (datatrove's
            ``language`` metadata).
        language_score: Its probability (datatrove's ``language_score``).
    """

    text: pa.ChunkedArray
    keep: pa.ChunkedArray
    reason: pa.ChunkedArray
    language: pa.ChunkedArray
    language_score: pa.ChunkedArray

    def __len__(self) -> int:
        return len(self.keep)


def _chunked(x: Any) -> pa.ChunkedArray:
    if isinstance(x, pa.ChunkedArray):
        return x
    if isinstance(x, pa.Array):
        return pa.chunked_array([x])
    if isinstance(x, (list, tuple)):
        return pa.chunked_array([pa.array(x, pa.string())])
    return pa.chunked_array(x)  # pyroid.Column, polars, ...: anything with __arrow_c_stream__


class FineWeb:
    """The FineWeb chain (see the module docstring), stage for stage as datatrove.

    Args:
        lid_model: A ``pyroid.fasttext.Model``, a path to ``lid.176.bin``, or
            ``None`` for :func:`lid_model_path`'s default (loaded on first use).
        languages: Languages to keep (``None``: any).
        language_threshold: datatrove ``LanguageFilter``'s ``language_threshold``.
        ftfy_config: A ``TextFixerConfig`` (``pyroid.ftfy``'s or ftfy's);
            ``None`` is ftfy's default config, as in the benchmark chain.
            ``False`` skips ftfy.
        filters: The quality filters, in order (``pyroid.filters`` configs or
            datatrove filter instances); ``None`` is the FineWeb set above.
        minhash: MinHash settings for :meth:`new_dedup` (a dict of
            ``MinhashConfig`` fields, or a ``MinhashConfig``); ``None`` disables dedup.
        threads: Default ``threads=`` for every stage (``None``: the whole pool).

    Language decisions are datatrove's: keep iff some requested language's
    probability is above the threshold. Above 0.501 that can only be the top
    label (fastText's probabilities sum to at most 1 + 176e-5), so the model is
    asked for its top label only; at lower thresholds it is asked for all labels.
    """

    name = "fineweb"

    def __init__(
        self,
        *,
        lid_model: Any = None,
        languages: Sequence[str] | None = ("en",),
        language_threshold: float = 0.65,
        ftfy_config: Any = None,
        filters: Sequence[Any] | None = None,
        minhash: Any = MINHASH_CONFIG,
        threads: int | None = None,
    ) -> None:
        self._model: Any = lid_model if isinstance(lid_model, pyroid.fasttext.Model) else None
        self._model_path = None if self._model is not None else lid_model
        self.languages = None if languages is None else [str(x) for x in languages]
        self.language_threshold = float(language_threshold)
        self.ftfy_config = ftfy_config
        self.filters = list(filters) if filters is not None else self.default_filters()
        self.stage_names = [_stage_name(f) for f in self.filters]
        self.minhash = minhash
        self.threads = threads

    @staticmethod
    def default_filters() -> list[Any]:
        """FineWeb's quality filters, in datatrove ``examples/fineweb.py``'s order."""
        return [
            pf.GopherRepetition(),
            pf.GopherQuality(),
            pf.C4Quality(filter_no_terminal_punct=False),
            pf.FineWebQuality(),
        ]

    @property
    def model(self) -> Any:
        """The ``pyroid.fasttext.Model`` (loaded on first use)."""
        if self._model is None:
            self._model = pyroid.fasttext.load_model(lid_model_path(self._model_path))
        return self._model

    def stages(self) -> list[str]:
        """Stage names, in order (``"null"`` is not a stage: it marks null texts)."""
        return ["lid", *self.stage_names, *(["minhash"] if self.minhash is not None else [])]

    def new_dedup(self, *, threads: int | None = None) -> pyroid.dedup.MinHashDedup | None:
        """A fresh MinHash index for this recipe's survivors (``None`` if dedup is off)."""
        if self.minhash is None:
            return None
        cfg = self.minhash
        t = self.threads if threads is None else threads
        if isinstance(cfg, dict):
            return pyroid.dedup.MinHashDedup(threads=t, **cfg)
        return pyroid.dedup.MinHashDedup(cfg, threads=t)

    # -------------------------------------------------------------------------------------- stages

    def fix(self, texts: Any, *, threads: int | None = None) -> pa.ChunkedArray:
        """The ftfy stage alone."""
        col = _chunked(texts)
        if self.ftfy_config is False:
            return col
        return _chunked(pyroid.ftfy.fix_text_batch(col, self.ftfy_config, threads=threads))

    def _lid(
        self, fixed: pa.ChunkedArray, threads: int | None
    ) -> tuple[list[bool], list[str | None], list[float | None]]:
        langs, thr = self.languages, self.language_threshold
        top_only = langs is None or thr > 0.501
        labels, probs = self.model.predict(
            pc.replace_substring(fixed, "\n", " "), k=1 if top_only else -1, threads=threads
        )
        keep: list[bool] = []
        lang: list[str | None] = []
        score: list[float | None] = []
        wanted = None if langs is None else {f"__label__{x}" for x in langs}
        for lb, p in zip(labels, probs, strict=True):
            if not lb:  # an Arrow null (fastText always predicts a label for a str)
                keep.append(False)
                lang.append(None)
                score.append(None)
                continue
            best, p0 = lb[0], float(p[0])
            lang.append(best.removeprefix("__label__"))
            score.append(p0)
            if wanted is None:
                keep.append(p0 > thr)
            elif top_only:
                keep.append(best in wanted and p0 > thr)
            else:
                keep.append(any(float(q) > thr for b, q in zip(lb, p, strict=True) if b in wanted))
        return keep, lang, score

    def _stage_of(self, reason: str, text: Any) -> str:
        """The stage that dropped a doc with ``reason``; re-runs prefixes only when ambiguous."""
        hits = [
            i
            for i, s in enumerate(self.stage_names)
            if (pat := _REASONS.get(s)) is None or pat.fullmatch(reason)
        ]
        if len(hits) == 1:
            return self.stage_names[hits[0]]
        for k in range(1, len(self.filters) + 1):
            if not pf.apply([text], self.filters[:k]).keep[0]:
                return self.stage_names[k - 1]
        raise AssertionError(f"no prefix of the filter chain drops this document ({reason!r})")

    def annotate(self, texts: Any, *, threads: int | None = None) -> Annotation:
        """Run ftfy, LID and the filters on one batch (see :class:`Annotation`).

        Args:
            texts: A list of str, or an Arrow string column (pyarrow, a
                ``pyroid.Column``, an HF dataset column, ...). Arrow nulls are
                dropped with reason ``"null"``.
            threads: Overrides the recipe's ``threads``.
        """
        t = self.threads if threads is None else threads
        fixed = self.fix(texts, threads=t)
        n = len(fixed)
        lid_keep, lang, score = self._lid(fixed, t)
        lid_mask = pa.array(lid_keep, pa.bool_())
        after_lid = fixed.filter(lid_mask)
        if self.filters:
            res = pf.apply(after_lid, self.filters, threads=t)
            f_keep = _chunked(res.keep).to_pylist()
            f_reason = _chunked(res.reason).to_pylist()
            f_text = _chunked(res.text)
        else:
            f_keep, f_reason, f_text = [True] * len(after_lid), [None] * len(after_lid), after_lid
        keep = [False] * n
        reason: list[str | None] = [None] * n
        fixed_null = fixed.is_null().to_pylist() if fixed.null_count else None
        j = 0
        for i in range(n):
            if not lid_keep[i]:
                reason[i] = "null" if fixed_null and fixed_null[i] else "lid"
                continue
            if f_keep[j]:
                keep[i] = True
            else:
                r = f_reason[j] or ""
                stage = self._stage_of(r, after_lid[j].as_py()) if self.filters else "filter"
                reason[i] = f"{stage}:{r}" if r else stage
            j += 1
        if len(f_text) == 0:
            text = fixed
        else:
            idx = pc.subtract(pc.cumulative_sum(lid_mask.cast(pa.int64())), 1)
            idx = pc.max_element_wise(idx, 0)
            text = pc.if_else(lid_mask, f_text.take(idx), fixed)
        return Annotation(
            text=_chunked(text),
            keep=pa.chunked_array([pa.array(keep, pa.bool_())]),
            reason=pa.chunked_array([pa.array(reason, pa.string())]),
            language=pa.chunked_array([pa.array(lang, pa.string())]),
            language_score=pa.chunked_array([pa.array(score, pa.float64())]),
        )


# ------------------------------------------------------------------------------ FineWeb from WARC

#: What datatrove 0.10.1's ``Trafilatura(favour_precision=True)`` step passes to
#: ``trafilatura.extract`` in ``examples/fineweb.py``, and the trafilatura release
#: ``datatrove[processing]==0.10.1`` installs (``compat``).
FINEWEB_TRAFILATURA: dict[str, Any] = {
    "compat": "1.11.0",
    "favor_precision": True,
    "include_comments": False,
}


class TaskExtractor:
    """datatrove's ``Trafilatura`` step over the pages of one task, on :mod:`pyroid.trafilatura`.

    Make one per datatrove task (in FineWeb, one per WARC file): it starts with an empty dedup
    cache, as the step's child process does. Called on a list of HTML pages, it returns for each
    the extracted text (``None`` or ``""`` when there is none: the step drops those), or the
    exception its extraction raised. datatrove's child process dies on such a page (the step
    counts ``broken_process`` and drops it) and the next child starts with an empty cache, so the
    cache is emptied after it here too. Pages are extracted in order: the dedup cache depends on
    it.

    Args:
        threads: ``threads=`` of :func:`pyroid.trafilatura.extract_batch` (``None``: the whole
            pool; with deduplication the pages are still extracted in order; threads parse
            ahead).
        **kwargs: ``trafilatura.extract`` arguments over :data:`FINEWEB_TRAFILATURA`. With
            ``compat="1.11.0"`` the dedup cache is :data:`pyroid.trafilatura.LRU_TEST`, as
            trafilatura 1.11.0 has only its module-level cache; with ``"2.3.1"``, an
            ``LRUCache`` of the task.
    """

    def __init__(self, *, threads: int | None = None, **kwargs: Any) -> None:
        from pyroid import _versions

        # compat picks the release, so only a trafilatura pyroid does not reproduce warns.
        with _versions.follow_installed("trafilatura"):
            import pyroid.trafilatura as pt

        self._pt: Any = pt
        self.kwargs = {**FINEWEB_TRAFILATURA, **kwargs}
        self.threads = threads
        self._module_cache = self.kwargs.get("compat") == "1.11.0"
        self.cache: Any = True
        self.reset()

    def reset(self) -> None:
        """Empty the dedup cache (a new datatrove child process)."""
        if self._module_cache:
            self._pt.reset_caches()
        else:
            self.cache = self._pt.LRUCache(maxsize=self._pt.LRU_SIZE)

    def _snapshot(self) -> Any:
        return copy.copy(self._pt.LRU_TEST if self._module_cache else self.cache)

    def _restore(self, snap: Any) -> None:
        if self._module_cache:
            self._pt.LRU_TEST = snap
        else:
            self.cache = snap

    def __call__(self, htmls: Sequence[str]) -> list[Any]:
        """The extracted text of each page (``str``/``None``), or the exception it raised."""
        pt = self._pt
        pages = list(htmls)
        snap = self._snapshot()
        try:
            return list(
                pt.extract_batch(pages, threads=self.threads, deduplicate=self.cache, **self.kwargs)
            )
        except Exception:  # redone page by page below
            self._restore(snap)
        # Redo the batch page by page from the snapshot: the page that raises is dropped and the
        # cache starts afresh after it, as datatrove's next child process does.
        out: list[Any] = []
        for page in pages:
            try:
                out.append(pt.extract(page, deduplicate=self.cache, **self.kwargs))
            except Exception as e:
                out.append(e)
                self.reset()
        return out


@dataclass(frozen=True)
class WarcAnnotation:
    """What :meth:`FineWebWarc.annotate` returns: one value per input document.

    Attributes:
        text: The text as it leaves the chain; for a dropped document, the text the dropping
            stage saw (the HTML for the URL filter and Trafilatura).
        keep: True if the document survived every stage but MinHash.
        reason: ``None`` if kept, else ``"url_filter:<reason>"``, ``"trafilatura"`` (no text
            extracted), ``"trafilatura:broken_process"`` (the extraction raised), ``"lid"`` or
            ``"<stage>:<datatrove reason>"``.
        language: fastText's top label (``None`` before the LID stage).
        language_score: Its probability.
    """

    text: list[str]
    keep: list[bool]
    reason: list[str | None]
    language: list[str | None]
    language_score: list[float | None]

    def __len__(self) -> int:
        return len(self.keep)


class FineWebWarc:
    """datatrove ``examples/fineweb.py`` from WARC files, stage for stage as datatrove 0.10.1.

    Per WARC file (one datatrove task each)::

        WarcReader (pyroid.warc.iter_documents: warcio's parsing, datatrove's process_record)
        -> URLFilter() -> Trafilatura(favour_precision=True), trafilatura 1.11.0
        -> LanguageFilter(languages=["en"], language_threshold=0.65)
        -> GopherRepetitionFilter() -> GopherQualityFilter()
        -> C4QualityFilter(filter_no_terminal_punct=False) -> FineWebQualityFilter()

    then MinHash dedup (:data:`MINHASH_CONFIG`) over the survivors of every file, in file
    order, and ``PIIFormatter()`` on what is left, a fresh rotation per file. :meth:`documents`,
    :meth:`annotate` (with a :meth:`new_task` extractor per file), :meth:`new_dedup` and
    :meth:`new_pii` are the pieces; ``pyroid clean --recipe fineweb-warc`` runs them.

    Args:
        lid_model: As :class:`FineWeb`'s.
        threads: Default ``threads=`` for every stage (``None``: the whole pool).
        url_filter: A :class:`pyroid.filters.URLFilter` (default: ``URLFilter()``, built at
            first use: it needs tldextract and datatrove's block lists).
        trafilatura: ``trafilatura.extract`` arguments over :data:`FINEWEB_TRAFILATURA`.
    """

    name = "fineweb-warc"

    def __init__(
        self,
        *,
        lid_model: Any = None,
        threads: int | None = None,
        url_filter: Any = None,
        trafilatura: dict[str, Any] | None = None,
    ) -> None:
        self._url_filter = url_filter
        self.fineweb = FineWeb(lid_model=lid_model, ftfy_config=False, threads=threads)
        self.trafilatura = {**FINEWEB_TRAFILATURA, **(trafilatura or {})}
        self.threads = threads
        self.minhash = MINHASH_CONFIG

    @property
    def url_filter(self) -> pf.URLFilter:
        """The :class:`pyroid.filters.URLFilter` (built on first use)."""
        if self._url_filter is None:
            self._url_filter = pf.URLFilter()
        return cast(pf.URLFilter, self._url_filter)

    def stages(self) -> list[str]:
        """Stage names, in order."""
        return ["url_filter", "trafilatura", *self.fineweb.stages(), "pii"]

    @staticmethod
    def documents(
        source: Any, *, compression: Literal["infer", "gzip"] | None = "infer", limit: int = -1
    ) -> Iterator[Document]:
        """The documents of one WARC file, as datatrove's ``WarcReader`` makes them.

        ``{"text", "id", "url", "date"}`` per document, from :func:`pyroid.warc.iter_documents`;
        documents with an empty text are skipped and at most ``limit`` are read (``-1``: all),
        as ``WarcReader`` does within one task.
        """
        import pyroid.warc as pw

        n = 0
        for _, doc in pw.iter_documents(source, compression=compression):
            if not doc.get("text"):
                continue
            if limit != -1 and n >= limit:
                return
            n += 1
            yield doc

    def new_task(self) -> TaskExtractor:
        """The Trafilatura step of one task (one per WARC file, in the file's order)."""
        return TaskExtractor(threads=self.threads, **self.trafilatura)

    def new_dedup(self, *, threads: int | None = None) -> pyroid.dedup.MinHashDedup | None:
        """A fresh MinHash index for the survivors of every file, in file order."""
        return self.fineweb.new_dedup(threads=self.threads if threads is None else threads)

    @staticmethod
    def new_pii() -> pf.PIIReplacer:
        """``PIIFormatter()`` for one file (one task): its replacement rotation starts at 0."""
        return pf.PIIReplacer()

    def annotate(
        self, docs: Sequence[Mapping[str, Any]], task: TaskExtractor, *, threads: int | None = None
    ) -> WarcAnnotation:
        """Run the URL filter, Trafilatura, LID and the quality filters on one batch.

        Args:
            docs: Documents of one file, in order (:meth:`documents`).
            task: That file's :meth:`new_task`.
            threads: Overrides the recipe's ``threads``.
        """
        t = self.threads if threads is None else threads
        n = len(docs)
        text: list[str] = [d["text"] for d in docs]
        keep = [False] * n
        reason: list[str | None] = [None] * n
        language: list[str | None] = [None] * n
        score: list[float | None] = [None] * n
        url_reasons = self.url_filter.reasons([d["url"] for d in docs], threads=t)
        live = [i for i, r in enumerate(url_reasons) if r is None]
        for i, r in enumerate(url_reasons):
            if r is not None:
                reason[i] = f"url_filter:{r}"
        extracted: list[int] = []
        for i, r in zip(live, task([text[i] for i in live]), strict=True):
            if isinstance(r, Exception):
                reason[i] = "trafilatura:broken_process"
            elif not r:
                reason[i] = "trafilatura"
            else:
                text[i] = r
                extracted.append(i)
        if extracted:
            ann = self.fineweb.annotate([text[i] for i in extracted], threads=t)
            columns = (
                ann.text.to_pylist(),
                ann.keep.to_pylist(),
                ann.reason.to_pylist(),
                ann.language.to_pylist(),
                ann.language_score.to_pylist(),
            )
            for i, txt, k, r, lg, sc in zip(extracted, *columns, strict=True):
                text[i], keep[i], reason[i], language[i], score[i] = txt, k, r, lg, sc
        return WarcAnnotation(text, keep, reason, language, score)
