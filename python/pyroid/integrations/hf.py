"""``Dataset.map`` and ``Dataset.filter`` helpers for Hugging Face ``datasets``.

Each helper returns a callable for ``batched=True``::

    import datasets
    from pyroid.integrations import hf

    ds = datasets.Dataset.from_parquet("fineweb.parquet").with_format("arrow")
    ds = ds.map(hf.fix_text(), batched=True)  # ftfy
    ds = ds.map(hf.langid(), batched=True)  # + language, language_score
    ds = ds.filter(hf.quality_mask(), batched=True)  # the Gopher, C4 and FineWeb filters

    res = hf.fineweb(datasets.Dataset.from_parquet("fineweb.parquet"))  # the whole recipe
    res.dataset.to_parquet("clean.parquet")
    print(res.stats)

The callables take both batch kinds ``datasets`` hands out:

- Arrow batches (``ds.with_format("arrow")``, the zero-copy path). The
  callable gets a ``pyarrow.Table`` whose text column pyroid reads in place,
  without building a Python ``str`` for it. A map callable returns a
  ``pyarrow.Table`` with every column of the batch. ``datasets`` replaces the
  batch with it, so ``remove_columns=`` has no effect: call
  ``.remove_columns()`` on the dataset. A filter callable returns a boolean
  ``pyarrow.ChunkedArray``.
- Python batches (the default format). The callable gets the dict of lists
  and returns a dict of the new or changed columns, which ``datasets`` merges
  into the batch, or a ``list[bool]`` for ``filter``. ``None`` texts stay
  ``None`` and are dropped by filters.

Arrow nulls are dropped by filters too (a null mask entry drops the row).
Don't pass ``input_columns=``: the callables read their column by name. They
work the same in a streaming ``IterableDataset``'s ``map`` and ``filter``,
except that an Arrow-format ``IterableDataset.filter`` needs datasets 5 (older
versions reject it whatever the function). :func:`fineweb` needs a
``Dataset``, because MinHash reads the survivors back. Tested with datasets
3.6, 4.8 and 5.1.

The callables pickle by their settings, so ``datasets`` can fingerprint them
(cached results are reused) and ship them to ``num_proc`` worker processes,
each of which loads the fastText model itself. A loaded
``pyroid.fasttext.Model`` can't be pickled: pass its path for both.

Needs pyarrow, numpy and datasets (``pip install 'pyroid[hf]'``).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from pyroid._deps import missing

try:
    import numpy  # noqa: F401  (pyroid.fasttext's arrays)
    import pyarrow as pa
    import pyarrow.compute as pc
except ModuleNotFoundError as _exc:
    missing(__name__, _exc, "hf", "pyarrow", "numpy")

import pyroid.fasttext
import pyroid.filters
import pyroid.ftfy
from pyroid.integrations.recipes import FineWeb, lid_model_path

__all__ = [
    "ANNOTATION_COLUMNS",
    "Batch",
    "Dataset",
    "FineWebResult",
    "LidModel",
    "Stats",
    "annotate",
    "fineweb",
    "fix_text",
    "keep_mask",
    "langid",
    "quality_mask",
]

#: A ``datasets.Dataset`` (``datasets`` ships no type information).
Dataset: TypeAlias = Any

#: A batch as ``datasets`` passes it: a ``pyarrow.Table`` or a dict of lists.
Batch: TypeAlias = Any

#: A fastText LID model: a path, a loaded ``pyroid.fasttext.Model``, or ``None``
#: (:func:`pyroid.integrations.recipes.lid_model_path`'s default).
LidModel: TypeAlias = "str | os.PathLike[str] | pyroid.fasttext.Model | None"

#: The columns :func:`annotate` adds (``language``/``language_score`` are
#: overwritten if present, as datatrove's ``LanguageFilter`` does).
ANNOTATION_COLUMNS = ("keep", "reason", "language", "language_score")


def _wrong_batch(batch: object) -> TypeError:
    return TypeError(
        f"expected a pyarrow.Table (ds.with_format('arrow')) or a dict of lists, got "
        f"{type(batch).__name__}; call Dataset.map/filter with batched=True and no input_columns"
    )


def _column(batch: Batch, name: str) -> Any:
    names = batch.column_names if isinstance(batch, pa.Table) else list(batch)
    if name not in names:
        raise KeyError(f"no column {name!r} in the batch (columns: {names})")
    return batch[name]


def _to_arrow(values: Sequence[Any]) -> Any:
    """A Python batch column as Arrow (``None`` -> null), for the helpers that return no text."""
    return pa.chunked_array([pa.array(values, pa.large_string())])


def _set(table: Any, name: str, values: Any) -> Any:
    """``table`` with column ``name`` replaced (in place) or appended (zero-copy)."""
    if isinstance(values, pa.Array):
        values = pa.chunked_array([values])
    elif not isinstance(values, pa.ChunkedArray):
        values = pa.chunked_array(values)  # a pyroid.Column, via __arrow_c_stream__
    i = table.schema.get_field_index(name)
    return table.set_column(i, name, values) if i >= 0 else table.append_column(name, values)


class _Helper:
    """Base class: dispatch on the batch kind; pickle by settings only."""

    _transient: tuple[str, ...] = ()

    def __call__(self, batch: Batch) -> Any:
        if isinstance(batch, pa.Table):
            return self._arrow(batch)
        if isinstance(batch, Mapping):
            return self._python(batch)
        raise _wrong_batch(batch)

    def _arrow(self, table: Any) -> Any:
        raise NotImplementedError

    def _python(self, batch: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def _settings(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k not in self._transient}

    def __getstate__(self) -> dict[str, Any]:
        state = self._settings()
        if isinstance(state.get("lid_model"), pyroid.fasttext.Model):
            raise TypeError(
                "a loaded pyroid.fasttext.Model can't be pickled: pass the model's path as "
                "lid_model so that datasets can cache the result and num_proc can ship it"
            )
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        for k in self._transient:
            setattr(self, k, None)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self._settings().items())
        return f"{type(self).__name__}({args})"


class _FixText(_Helper):
    def __init__(
        self, column: str, config: Any, kwargs: dict[str, Any], threads: int | None
    ) -> None:
        self.column, self.config, self.kwargs, self.threads = column, config, kwargs, threads

    def _fix(self, texts: Any) -> Any:
        return pyroid.ftfy.fix_text_batch(texts, self.config, threads=self.threads, **self.kwargs)

    def _arrow(self, table: Any) -> Any:
        return _set(table, self.column, self._fix(_column(table, self.column)))

    def _python(self, batch: Mapping[str, Any]) -> dict[str, list[Any]]:
        values = list(_column(batch, self.column))
        pos = [i for i, v in enumerate(values) if v is not None]
        if len(pos) == len(values):
            return {self.column: self._fix(values)}
        out = list(values)
        for i, v in zip(pos, self._fix([values[i] for i in pos]), strict=True):
            out[i] = v
        return {self.column: out}


def fix_text(
    column: str = "text",
    config: pyroid.ftfy.TextFixerConfig | None = None,
    *,
    threads: int | None = None,
    **kwargs: Any,
) -> Callable[[Batch], Any]:
    """A ``Dataset.map(batched=True)`` function that runs ``ftfy.fix_text`` on a column.

    Args:
        column: The text column, replaced by its fixed text.
        config: As for :func:`pyroid.ftfy.fix_text_batch` (``None``: ftfy's defaults).
        threads: pyroid's ``threads=`` (``None``: the whole thread pool).
        **kwargs: Config fields, as for :func:`pyroid.ftfy.fix_text_batch`.

    Returns:
        The map function (see the module docstring for the batch kinds).
    """
    return _FixText(column, config, kwargs, threads)


class _LangID(_Helper):
    _transient = ("_model",)

    def __init__(
        self,
        column: str,
        lid_model: Any,
        k: int,
        threads: int | None,
        language_column: str,
        score_column: str,
    ) -> None:
        self.column, self.lid_model, self.k, self.threads = column, lid_model, k, threads
        self.language_column, self.score_column = language_column, score_column
        self._model: pyroid.fasttext.Model | None = None

    @property
    def model(self) -> pyroid.fasttext.Model:
        if self._model is None:
            m = self.lid_model
            self._model = (
                m
                if isinstance(m, pyroid.fasttext.Model)
                else pyroid.fasttext.load_model(lid_model_path(m))
            )
        return self._model

    def _predict(self, texts: Any) -> tuple[Any, Any]:
        # datatrove's FastTextLID.predict: one line per document.
        labels, probs = self.model.predict(
            pc.replace_substring(texts, "\n", " "), k=self.k, threads=self.threads
        )
        if self.k == 1:
            lang = [lb[0].removeprefix("__label__") if lb else None for lb in labels]
            score = [float(p[0]) if lb else None for lb, p in zip(labels, probs, strict=True)]
            return pa.array(lang, pa.string()), pa.array(score, pa.float64())
        langs = [[x.removeprefix("__label__") for x in lb] if lb else None for lb in labels]
        scores = [
            [float(x) for x in p] if lb else None for lb, p in zip(labels, probs, strict=True)
        ]
        return pa.array(langs, pa.list_(pa.string())), pa.array(scores, pa.list_(pa.float64()))

    def _arrow(self, table: Any) -> Any:
        lang, score = self._predict(_column(table, self.column))
        return _set(_set(table, self.language_column, lang), self.score_column, score)

    def _python(self, batch: Mapping[str, Any]) -> dict[str, list[Any]]:
        lang, score = self._predict(_to_arrow(_column(batch, self.column)))
        return {self.language_column: lang.to_pylist(), self.score_column: score.to_pylist()}


def langid(
    lid_model: LidModel = None,
    column: str = "text",
    *,
    k: int = 1,
    threads: int | None = None,
    language_column: str = "language",
    score_column: str = "language_score",
) -> Callable[[Batch], Any]:
    """A ``Dataset.map(batched=True)`` function that adds fastText language ID columns.

    Each document is predicted as one line (newlines become spaces, as in
    datatrove's ``FastTextLID``).

    Args:
        lid_model: A path to a fastText LID model (``lid.176.bin``), a loaded
            ``pyroid.fasttext.Model`` (in-process only, see the module
            docstring), or ``None`` for :func:`pyroid.integrations.recipes.lid_model_path`
            (``$PYROID_LID_MODEL``, else downloaded once into a cache).
        column: The text column.
        k: Number of labels; ``1`` adds a string and a float column, any other
            value (``-1``: all labels) adds list columns, best first.
        threads: pyroid's ``threads=``.
        language_column: Name of the label column (labels without ``__label__``).
        score_column: Name of the probability column (float64).

    Returns:
        The map function. Null texts get null labels.
    """
    return _LangID(column, lid_model, k, threads, language_column, score_column)


class _QualityMask(_Helper):
    def __init__(self, column: str, filters: list[Any] | None, threads: int | None) -> None:
        self.column, self.filters, self.threads = column, filters, threads

    def _keep(self, texts: Any) -> Any:
        filters = self.filters if self.filters is not None else FineWeb.default_filters()
        return pa.chunked_array(pyroid.filters.apply(texts, filters, threads=self.threads).keep)

    def _arrow(self, table: Any) -> Any:
        return self._keep(_column(table, self.column))

    def _python(self, batch: Mapping[str, Any]) -> list[bool]:
        return [bool(k) for k in self._keep(_to_arrow(_column(batch, self.column))).to_pylist()]


def quality_mask(
    column: str = "text",
    filters: Sequence[Any] | None = None,
    *,
    threads: int | None = None,
) -> Callable[[Batch], Any]:
    """A ``Dataset.filter(batched=True)`` function: keep what passes the quality filters.

    The filters run in order, as datatrove's pipeline does
    (:func:`pyroid.filters.apply`). Only the keep decision is used:
    ``C4Quality``'s line removal is not applied to the text (use
    :func:`annotate` or :func:`fineweb` for the text as it leaves the chain).

    Args:
        column: The text column.
        filters: ``pyroid.filters`` configs or datatrove filter instances, in
            order; ``None`` is FineWeb's set (GopherRepetition, GopherQuality,
            C4Quality(filter_no_terminal_punct=False), FineWebQuality).
        threads: pyroid's ``threads=``.

    Returns:
        The filter function.
    """
    return _QualityMask(column, None if filters is None else list(filters), threads)


class _KeepMask(_Helper):
    def __init__(self, column: str) -> None:
        self.column = column

    def _arrow(self, table: Any) -> Any:
        return _column(table, self.column)

    def _python(self, batch: Mapping[str, Any]) -> list[bool]:
        return [bool(k) for k in _column(batch, self.column)]


def keep_mask(column: str = "keep") -> Callable[[Batch], Any]:
    """A ``Dataset.filter(batched=True)`` function that reads a boolean column.

    For instance the ``keep`` column of :func:`annotate`
    (``ds.filter(hf.keep_mask(), batched=True)``). Nulls drop the row.

    Args:
        column: The boolean column.

    Returns:
        The filter function.
    """
    return _KeepMask(column)


class _Annotate(_Helper):
    _transient = ("_recipe",)

    def __init__(self, column: str, lid_model: Any, threads: int | None) -> None:
        self.column, self.lid_model, self.threads = column, lid_model, threads
        self._recipe: FineWeb | None = None

    @property
    def recipe(self) -> FineWeb:
        if self._recipe is None:
            self._recipe = FineWeb(lid_model=self.lid_model, threads=self.threads)
        return self._recipe

    def _arrow(self, table: Any) -> Any:
        a = self.recipe.annotate(_column(table, self.column))
        table = _set(table, self.column, a.text)
        for name in ANNOTATION_COLUMNS:
            table = _set(table, name, getattr(a, name))
        return table

    def _python(self, batch: Mapping[str, Any]) -> dict[str, list[Any]]:
        a = self.recipe.annotate(_to_arrow(_column(batch, self.column)))
        return {
            self.column: a.text.to_pylist(),
            **{name: getattr(a, name).to_pylist() for name in ANNOTATION_COLUMNS},
        }


def annotate(
    column: str = "text",
    *,
    lid_model: LidModel = None,
    threads: int | None = None,
) -> Callable[[Batch], Any]:
    """A ``Dataset.map(batched=True)`` function that runs the FineWeb chain but the dedup.

    Per batch, :meth:`pyroid.integrations.recipes.FineWeb.annotate`: ftfy ->
    LID (en, p > 0.65) -> GopherRepetition -> GopherQuality -> C4Quality ->
    FineWebQuality. ``column`` becomes the text as it leaves the chain (for a
    dropped document, the text the dropping stage saw) and these columns are
    set: ``keep`` (bool), ``reason`` (``None`` if kept, else ``"lid"``,
    ``"<stage>:<datatrove reason>"`` or ``"null"``), ``language`` and
    ``language_score``. Drop with ``ds.filter(hf.keep_mask(), batched=True)``;
    MinHash dedup needs the whole corpus: see :func:`fineweb`.

    Args:
        column: The text column.
        lid_model: As for :func:`langid`.
        threads: pyroid's ``threads=``.

    Returns:
        The map function.
    """
    return _Annotate(column, lid_model, threads)


@dataclass(frozen=True)
class Stats:
    """What :func:`fineweb` did.

    Attributes:
        docs: Documents in.
        kept: Documents out.
        dropped: Documents dropped per stage, in chain order (``lid``, the
            four filters, ``minhash``; ``null`` too if there were null texts).
        reasons: Documents dropped per reason (``"gopher_quality:gopher_short_doc"``,
            ``"minhash"``, ...), most frequent first.
        seconds: Wall time: ``annotate`` (the ``Dataset.map`` of ftfy, LID and the
            filters), ``dedup`` (MinHash over the survivors), ``total`` (including
            building the output datasets).
    """

    docs: int
    kept: int
    dropped: dict[str, int] = field(default_factory=dict)
    reasons: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        pct = 100 * self.kept / self.docs if self.docs else 0.0
        lines = [f"{self.docs:,} docs -> {self.kept:,} kept ({pct:.1f}%)"]
        lines += [f"  dropped by {stage:<18} {n:>8,}" for stage, n in self.dropped.items()]
        if self.seconds:
            lines.append("  " + ", ".join(f"{k} {v:.2f} s" for k, v in self.seconds.items()))
        return "\n".join(lines)


@dataclass(frozen=True)
class FineWebResult:
    """What :func:`fineweb` returns.

    Attributes:
        dataset: The kept documents, in order: cleaned text, the input's other
            columns, and ``language``/``language_score``.
        annotated: Every document, with the text as it left the chain and the
            columns ``keep`` (the final decision) and ``reason`` (``None`` if
            kept; ``"minhash"`` for a near-duplicate; see :func:`annotate`).
        stats: Counts and timings.
    """

    dataset: Dataset
    annotated: Dataset
    stats: Stats


def fineweb(
    ds: Dataset,
    column: str = "text",
    *,
    lid_model: LidModel = None,
    threads: int | None = None,
    batch_size: int = 1000,
    num_proc: int | None = None,
    **map_kwargs: Any,
) -> FineWebResult:
    """Run the FineWeb recipe on a dataset, MinHash dedup included.

    1. ``ds.map(annotate(...), batched=True)`` on Arrow batches: ftfy, LID and
       the four quality filters (:func:`annotate`).
    2. MinHash (datatrove's FineWeb settings) over the survivors, read back in
       dataset order: each cluster of near-duplicates keeps one document, as in
       datatrove, and the others are dropped (``reason="minhash"``). This step
       always runs in this process over the whole dataset, so it is global and
       its result does not depend on ``batch_size`` or ``num_proc``.

    On the same documents, the result equals the same chain built from datatrove's stages
    (``bench/chain.py``'s digest): ftfy with ftfy's default config, then the stages of
    datatrove's FineWeb pipeline that work on text.

    Args:
        ds: A ``datasets.Dataset`` (any format; the results get the same format type).
        column: The text column.
        lid_model: As for :func:`langid`.
        threads: pyroid's ``threads=`` for every stage. With ``num_proc`` worker
            processes, about ``os.cpu_count() // num_proc`` avoids oversubscription.
        batch_size: Documents per ``Dataset.map`` batch (and per dedup read).
        num_proc: ``Dataset.map``'s worker processes for step 1 (each loads the
            fastText model). pyroid already runs every batch on all cores, so this
            rarely helps; ``lid_model`` must be a path (or ``None``).
        **map_kwargs: Passed to step 1's ``Dataset.map`` (``keep_in_memory``,
            ``load_from_cache_file``, ``cache_file_name``, ``desc``, ...).

    Returns:
        A :class:`FineWebResult`.

    Raises:
        ValueError: ``ds`` is empty, or ``num_proc > 1`` with a loaded
            ``pyroid.fasttext.Model``.
    """
    if (num_proc or 0) > 1 and isinstance(lid_model, pyroid.fasttext.Model):
        raise ValueError("num_proc > 1 needs lid_model as a path: a loaded Model can't be pickled")
    if len(ds) == 0:
        raise ValueError("the dataset is empty")
    t0 = time.perf_counter()
    fmt = ds.format
    map_kwargs.setdefault("desc", "pyroid FineWeb")
    annotated = ds.with_format("arrow").map(
        annotate(column, lid_model=lid_model, threads=threads),
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        **map_kwargs,
    )
    t1 = time.perf_counter()

    recipe = FineWeb(lid_model=lid_model, threads=threads)  # the model is not loaded here
    dedup = recipe.new_dedup()
    assert dedup is not None
    for batch in annotated.iter(batch_size=batch_size):
        dedup.add(batch[column].filter(batch["keep"]))
    dup_of_survivor = pa.array([not k for k in dedup.keep_mask()], pa.bool_())
    t2 = time.perf_counter()

    # Scatter the survivors' dedup decisions back to every document.
    survived = annotated["keep"].combine_chunks()
    if len(dup_of_survivor):
        nth = pc.max_element_wise(pc.subtract(pc.cumulative_sum(survived.cast(pa.int64())), 1), 0)
        dup = pc.and_(survived, dup_of_survivor.take(nth))
    else:
        dup = pa.repeat(pa.scalar(False), len(survived))
    keep = pc.and_(survived, pc.invert(dup))
    reason = pc.if_else(
        dup, pa.scalar("minhash", pa.string()), annotated["reason"].combine_chunks()
    )
    annotated = (
        annotated.remove_columns(["keep", "reason"])
        .add_column("keep", keep)
        .add_column("reason", reason)
    )
    # int64: datasets 5.1's contiguous-select fast path fails on uint64 indices.
    kept_rows = pc.indices_nonzero(keep).cast(pa.int64()).to_numpy()
    dataset = annotated.select(kept_rows).remove_columns(["keep", "reason"])

    counts = pc.value_counts(pc.drop_null(reason)).to_pylist()
    reasons = {
        c["values"]: c["counts"] for c in sorted(counts, key=lambda c: (-c["counts"], c["values"]))
    }
    dropped = dict.fromkeys(recipe.stages(), 0)
    for r, n in reasons.items():
        stage = r.split(":", 1)[0]
        dropped[stage] = dropped.get(stage, 0) + n
    stats = Stats(
        docs=len(keep),
        kept=len(dataset),
        dropped=dropped,
        reasons=reasons,
        seconds={
            "annotate": t1 - t0,
            "dedup": t2 - t1,
            "total": time.perf_counter() - t0,
        },
    )
    restore = {"type": fmt["type"], **fmt["format_kwargs"]}
    return FineWebResult(
        dataset=dataset.with_format(**restore),
        annotated=annotated.with_format(**restore),
        stats=stats,
    )
