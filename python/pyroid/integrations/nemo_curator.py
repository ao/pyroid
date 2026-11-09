# Derived from NeMo Curator 1.4.0 (Apache-2.0), © NVIDIA CORPORATION; modified.
# mypy: disallow-subclassing-any=False
# (NeMo Curator's classes are Any to mypy where it is not installed.)
"""Drop-in NeMo Curator text stages backed by pyroid.

Change the imports and leave the pipeline as it is::

    # from nemo_curator.stages.text.filters import ScoreFilter
    # from nemo_curator.stages.text.filters.heuristic import WordCountFilter
    from pyroid.integrations.nemo_curator import ScoreFilter, WordCountFilter

    pipeline.add_stage(ScoreFilter(WordCountFilter(min_words=80)))

The stages :class:`ScoreFilter`, :class:`Score`, :class:`Filter` and :class:`Modify` subclass
NeMo's and take the same arguments, plus a keyword ``threads`` (pyroid's ``threads=``, default 1:
NeMo already runs one worker per CPU). They score or modify a whole batch in one pyroid call instead
of calling ``score_document`` / ``modify_document`` once per row through ``pandas.Series.apply``,
and they do this for NeMo's own filter and modifier objects too, so a pipeline built from NeMo's
YAML configs only needs its ``ScoreFilter`` / ``Modify`` targets changed (or
:func:`pyroid.patch` with ``"nemo_curator"``).

The filters and modifiers here subclass NeMo's classes of the same name and take the same
arguments. Their ``score_document`` / ``modify_document`` run pyroid's kernel for one document, so
they are faster inside NeMo's own stages as well:

- the heuristic filters of ``nemo_curator.stages.text.filters.heuristic`` (string and repetition);
- :class:`FastTextLangId` and :class:`FastTextQualityFilter`, which load their model with
  :mod:`pyroid.fasttext` (``fasttext``, which needs a C++ compiler to install, is not needed);
- :class:`UnicodeReformatter` (:mod:`pyroid.ftfy`; ftfy is not needed), and the string
  modifiers :class:`BoilerPlateStringModifier`, :class:`MarkdownRemover`,
  :class:`NewlineNormalizer`, :class:`QuotationRemover`, :class:`UrlRemover`,
  :class:`LineRemover`, :class:`Slicer` and :class:`FastTextLabelModifier`.

The output is the same as with NeMo's code: the same scores (same values and pandas dtypes), the
same kept rows, the same text, and the same exception where NeMo raises one (for a document NeMo's
code would fail on, the stage runs NeMo's code, which raises). Anything pyroid does not port runs
NeMo's own code, with a one-time :class:`pyroid.FallbackWarning` where a setting caused it: a
custom ``url_regex``, Japanese word splitting (MeCab), precomputed ``sentences`` / ``paragraphs``
/ ``ngrams`` on a filter, subclasses of these classes, and multi-input modifiers.

``RepeatingTopNGramsFilter`` breaks ties between equally frequent n-grams the way NeMo does: by
the order of a Python ``set``, which depends on the process's string hash seed. NeMo's own scores
for such documents therefore change from run to run unless ``PYTHONHASHSEED`` is fixed; pyroid
gives the score NeMo gives in the same process.

Needs NeMo Curator 1.4.0 (Python 3.11 to 3.13); :mod:`pyroid.nemo` has the same kernels without it.
"""

from __future__ import annotations

import ast
import dataclasses
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
from nemo_curator.stages import base as _nc_base
from nemo_curator.stages.text.filters import doc_filter as _nc_doc_filter
from nemo_curator.stages.text.filters import score_filter as _nc_sf
from nemo_curator.stages.text.filters.heuristic import string as _nc_hs
from nemo_curator.stages.text.filters.heuristic.repetition import repetition as _nc_hr
from nemo_curator.stages.text.modifiers import doc_modifier as _nc_doc_modifier
from nemo_curator.stages.text.modifiers import modifier as _nc_modifier
from nemo_curator.stages.text.modifiers import string as _nc_ms
from nemo_curator.stages.text.modifiers.fasttext import fasttext_label as _nc_ftl
from nemo_curator.stages.text.utils import constants as _nc_constants

import pyroid
import pyroid.ftfy
import pyroid.nemo as _k

__all__ = [
    "BoilerPlateStringFilter",
    "BoilerPlateStringModifier",
    "BulletsFilter",
    "CommonEnglishWordsFilter",
    "EllipsisFilter",
    "FastTextLabelModifier",
    "FastTextLangId",
    "FastTextQualityFilter",
    "Filter",
    "LineRemover",
    "LongWordFilter",
    "MarkdownRemover",
    "MeanWordLengthFilter",
    "Modify",
    "NewlineNormalizer",
    "NonAlphaNumericFilter",
    "NumbersFilter",
    "ParenthesesFilter",
    "PornographicUrlsFilter",
    "PunctuationFilter",
    "QuotationRemover",
    "RepeatedLinesByCharFilter",
    "RepeatedLinesFilter",
    "RepeatedParagraphsByCharFilter",
    "RepeatedParagraphsFilter",
    "RepeatingDuplicateNGramsFilter",
    "RepeatingTopNGramsFilter",
    "Score",
    "ScoreFilter",
    "Slicer",
    "SubstringFilter",
    "SymbolsToWordsFilter",
    "UnicodeReformatter",
    "UrlRemover",
    "UrlsFilter",
    "WhiteSpaceFilter",
    "WordCountFilter",
    "WordsWithoutAlphabetsFilter",
]

#: The NeMo Curator release whose stages this module reproduces.
NEMO_CURATOR_VERSION = _k.NEMO_CURATOR_VERSION

_WARNED: set[str] = set()

# Errors after which a batch (or a document) is run through NeMo's own code instead: either NeMo
# raises the same way (empty documents, missing text, ...), or pyroid does not handle the input.
_FALLBACK_ERRORS = (
    ArithmeticError,
    LookupError,
    TypeError,
    ValueError,
    UnicodeError,
    pyroid.PyroidError,
)


def _check_nemo_version() -> None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("nemo_curator")
    except PackageNotFoundError:  # pragma: no cover - imported from a source tree
        return
    if installed != _k.NEMO_CURATOR_VERSION:
        warnings.warn(
            f"pyroid.integrations.nemo_curator reproduces NeMo Curator {_k.NEMO_CURATOR_VERSION}; "
            f"NeMo Curator {installed} is installed, whose own stages may differ",
            RuntimeWarning,
            stacklevel=3,
        )


_check_nemo_version()


def _fallback(what: str, reason: str) -> None:
    key = f"{what}:{reason}"
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(
            f"pyroid.integrations.nemo_curator.{what}: {reason} is not supported by pyroid; "
            "running NeMo Curator's own code (slow path). This warning is shown once.",
            pyroid.FallbackWarning,
            stacklevel=4,
        )


# --- batches: pandas Series <-> pyroid input --------------------------------------------------


def _texts(series: pd.Series) -> Any:
    """``series`` as pyroid input (an Arrow array or a list of str), or None if it can't be."""
    arr = series.array
    if hasattr(arr, "__arrow_array__"):
        import pyarrow as pa

        chunked = arr.__arrow_array__()
        t = chunked.type
        if not (pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_string_view(t)):
            return None
        if chunked.null_count:
            # NeMo's code fails on missing text; let it.
            return None
        return chunked
    if series.dtype == object or isinstance(series.dtype, pd.StringDtype):
        return series.tolist()
    return None


def _like_apply(series: pd.Series, values: Any) -> pd.Series:
    """``values`` as the Series ``series.apply(f)`` gives when ``f`` returns them (dtype, index)."""
    sample = values[0]
    if isinstance(sample, np.generic):
        sample = sample.item()
    elif hasattr(sample, "as_py"):  # a pyarrow scalar
        sample = sample.as_py()
    # pandas infers the result dtype from the values and the input's dtype (Arrow-backed input
    # gives Arrow-backed results); ask it, with one call on one row.
    dtype = series.iloc[:1].apply(lambda _: sample).dtype
    if isinstance(values, np.ndarray):
        out = pd.Series(values, index=series.index, name=series.name)
        return out if out.dtype == dtype else out.astype(dtype)
    return pd.Series(values, index=series.index, name=series.name, dtype=dtype)


# --- filters: what pyroid computes for each NeMo DocumentFilter --------------------------------


class _Plan(NamedTuple):
    """How pyroid scores (and keeps) for one filter object, from its current attributes."""

    score: Callable[[Any, int | None], Any]
    keep: Callable[[np.ndarray], np.ndarray] | None


def _is_number(x: object) -> bool:
    if type(x) is float or type(x) is bool:
        return True
    return type(x) is int and abs(x) < 2**53


_SPLITTER_LANG = {"default_splitter": "en", "jieba_splitter": "zh", "mecab_splitter": "ja"}


def _lang(f: Any) -> str | None:
    """The language of ``f``'s word splitter (``get_word_splitter``), or None if not NeMo's."""
    splitter = getattr(f, "_word_splitter", None)
    if getattr(splitter, "__module__", None) != "nemo_curator.stages.text.utils.text_utils":
        return None
    return _SPLITTER_LANG.get(getattr(splitter, "__name__", ""))


def _le(attr: str) -> Callable[[Any], Callable[[np.ndarray], np.ndarray] | None]:
    def make(f: Any) -> Callable[[np.ndarray], np.ndarray] | None:
        c = getattr(f, attr)
        return (lambda s: s <= c) if _is_number(c) else None

    return make


def _ge(attr: str) -> Callable[[Any], Callable[[np.ndarray], np.ndarray] | None]:
    def make(f: Any) -> Callable[[np.ndarray], np.ndarray] | None:
        c = getattr(f, attr)
        return (lambda s: s >= c) if _is_number(c) else None

    return make


def _between(
    lo_attr: str, hi_attr: str
) -> Callable[[Any], Callable[[np.ndarray], np.ndarray] | None]:
    def make(f: Any) -> Callable[[np.ndarray], np.ndarray] | None:
        lo, hi = getattr(f, lo_attr), getattr(f, hi_attr)
        if not (_is_number(lo) and _is_number(hi)):
            return None
        return lambda s: (lo <= s) & (s <= hi)

    return make


def _no_state(f: Any, *attrs: str) -> bool:
    return all(getattr(f, a, None) is None for a in attrs)


_Adapter = Callable[[Any], "_Plan | str | None"]


def _simple(fn: Callable[..., Any], keep: Callable[[Any], Any], *state: str) -> _Adapter:
    """A filter whose score is ``fn(texts)`` and whose state attributes ``state`` must be unset."""

    def adapt(f: Any) -> _Plan | str | None:
        if not _no_state(f, *state):
            return "precomputed sentences, paragraphs or n-grams"
        return _Plan(lambda x, t: fn(x, threads=t), keep(f))

    return adapt


def _with_lang(fn: Callable[..., Any], keep: Callable[[Any], Any], *state: str) -> _Adapter:
    def adapt(f: Any) -> _Plan | str | None:
        if not _no_state(f, *state):
            return "precomputed sentences, paragraphs or n-grams"
        lang = _lang(f)
        if lang is None:
            return "a custom word splitter"
        if lang == "ja":
            return "Japanese word splitting (MeCab)"
        return _Plan(lambda x, t: fn(x, lang=lang, threads=t), keep(f))

    return adapt


def _default_url_regex(fn: Callable[..., Any], keep: Callable[[Any], Any]) -> _Adapter:
    def adapt(f: Any) -> _Plan | str | None:
        if f._url_regex is not _nc_constants.regex_url:
            return "a custom url_regex"
        return _Plan(lambda x, t: fn(x, threads=t), keep(f))

    return adapt


def _boilerplate(f: Any) -> _Plan | str | None:
    if type(f._max_ratio) is not float or f._max_ratio != 1.0:
        return "a changed _max_ratio"
    return _Plan(
        lambda x, t: _k.boilerplate_string(x, threads=t), _le("_max_boilerplate_string_ratio")(f)
    )


def _common_english(f: Any) -> _Plan | str | None:
    if _lang(f) != "en":
        return "a custom word splitter"
    cutoff, stop = f._cutoff, f._stop_at_false
    if type(cutoff) is not int or abs(cutoff) >= 2**53:
        return f"min_num_common_words={cutoff!r}"
    return _Plan(
        lambda x, t: _k.common_english_words(
            x, min_num_common_words=cutoff, stop_at_false=bool(stop), threads=t
        ),
        lambda s: s >= cutoff,
    )


def _ngrams(fn: Callable[..., Any]) -> _Adapter:
    def adapt(f: Any) -> _Plan | str | None:
        if f._ngrams is not None:
            return "precomputed n-grams"
        if type(f._max_ratio) is not float or f._max_ratio != 1.0:
            return "a changed _max_ratio"
        n = f._n
        if type(n) is not int or n < 1:
            return f"n={n!r}"
        lang = _lang(f)
        if lang is None:
            return "a custom word splitter"
        if lang == "ja":
            return "Japanese word splitting (MeCab)"
        return _Plan(lambda x, t: fn(x, n, lang=lang, threads=t), _le("_cutoff")(f))

    return adapt


def _substring(f: Any) -> _Plan | str | None:
    sub, pos = f._substring, f._position
    if type(sub) is not str or pos not in ("prefix", "suffix", "any"):
        return "a non-str substring"
    return _Plan(lambda x, t: _k.substring(x, sub, pos, threads=t), lambda s: s == 1)


_FILTERS: dict[type, _Adapter] = {
    _nc_hs.NonAlphaNumericFilter: _simple(_k.non_alpha_numeric, _le("_cutoff")),
    _nc_hs.SymbolsToWordsFilter: _with_lang(_k.symbols_to_words, _le("_cutoff")),
    _nc_hs.NumbersFilter: _simple(_k.numbers, _le("_cutoff")),
    _nc_hs.UrlsFilter: _default_url_regex(_k.urls, _le("_cutoff")),
    _nc_hs.BulletsFilter: _simple(_k.bullets, _le("_cutoff"), "_sentences"),
    _nc_hs.WhiteSpaceFilter: _simple(_k.white_space, _le("_cutoff")),
    _nc_hs.ParenthesesFilter: _simple(_k.parentheses, _le("_max_parentheses_ratio")),
    _nc_hs.LongWordFilter: _with_lang(_k.long_word, _le("_max_word_length")),
    _nc_hs.WordCountFilter: _with_lang(_k.word_count, _between("_min_words", "_max_words")),
    _nc_hs.BoilerPlateStringFilter: _boilerplate,
    _nc_hs.MeanWordLengthFilter: _with_lang(
        _k.mean_word_length, _between("_min_cutoff", "_max_cutoff")
    ),
    _nc_hs.PunctuationFilter: _simple(_k.punctuation, _le("_cutoff"), "_sentences"),
    _nc_hs.EllipsisFilter: _simple(_k.ellipsis, _le("_cutoff"), "_sentences"),
    _nc_hs.CommonEnglishWordsFilter: _common_english,
    _nc_hs.WordsWithoutAlphabetsFilter: _with_lang(_k.words_without_alphabets, _ge("_cutoff")),
    _nc_hs.PornographicUrlsFilter: _default_url_regex(
        _k.pornographic_urls, lambda f: lambda s: s != 1
    ),
    _nc_hs.SubstringFilter: _substring,
    _nc_hr.RepeatedLinesFilter: _simple(_k.repeated_lines, _ge("_cutoff"), "_sentences"),
    _nc_hr.RepeatedParagraphsFilter: _simple(
        _k.repeated_paragraphs, _ge("_max_repeated_paragraphs_ratio"), "_paragraphs"
    ),
    _nc_hr.RepeatedLinesByCharFilter: _simple(
        _k.repeated_lines_by_char, _ge("_cutoff"), "_sentences"
    ),
    _nc_hr.RepeatedParagraphsByCharFilter: _simple(
        _k.repeated_paragraphs_by_char, _ge("_cutoff"), "_paragraphs"
    ),
    _nc_hr.RepeatingTopNGramsFilter: _ngrams(_k.repeating_top_ngrams),
    _nc_hr.RepeatingDuplicateNGramsFilter: _ngrams(_k.repeating_duplicate_ngrams),
}


def _plan(f: Any) -> _Plan | None:
    """The kernel plan for filter ``f``, or None (with a one-time warning if a setting is why)."""
    adapt = _FILTERS.get(type(f))
    if adapt is None:
        return None
    plan = adapt(f)
    if isinstance(plan, str):
        _fallback(type(f).__name__, plan)
        return None
    return plan


def _score_batch(f: Any, series: pd.Series, threads: int | None) -> pd.Series | None:
    """``series.apply(f.score_document)`` in one pyroid call, or None to run NeMo's code instead."""
    plan = _plan(f)
    if plan is None or len(series) == 0:
        return None
    texts = _texts(series)
    if texts is None:
        return None
    try:
        values = plan.score(texts, threads)
    except NotImplementedError as e:
        _fallback(type(f).__name__, str(e))
        return None
    except _FALLBACK_ERRORS:
        return None
    return _like_apply(series, values)


def _keep_batch(f: Any, scores: pd.Series) -> np.ndarray | None:
    """``scores.apply(f.keep_document)`` as a bool array, or None to run NeMo's code instead."""
    plan = _plan(f)
    if plan is None or plan.keep is None or len(scores) == 0 or scores.hasnans:
        return None
    try:
        values = scores.to_numpy()
    except (TypeError, ValueError):
        return None
    if values.dtype.kind not in "biuf":
        return None
    return np.asarray(plan.keep(values), dtype=bool)


class _OneDoc:
    """``score_document`` through pyroid for one document, NeMo's code otherwise."""

    def score_document(self, text: str) -> Any:
        plan = _plan(self)
        if plan is not None and type(text) is str:
            try:
                return plan.score(text, 1)
            except NotImplementedError as e:
                _fallback(type(self).__name__, str(e))
            except _FALLBACK_ERRORS:
                pass
        return super().score_document(text)  # type: ignore[misc]


# --- fastText -------------------------------------------------------------------------------------

try:
    from nemo_curator.stages.text.filters.fasttext import fasttext_filters as _nc_ft

    _LangIdBase: Any = _nc_ft.FastTextLangId
    _QualityBase: Any = _nc_ft.FastTextQualityFilter
except ImportError:  # fasttext is not installed: NeMo's module imports it at the top
    _nc_ft = None
    _LangIdBase = _QualityBase = _nc_doc_filter.DocumentFilter


class FastTextQualityFilter(_QualityBase):
    """NeMo's ``FastTextQualityFilter``, with the model loaded by :mod:`pyroid.fasttext`.

    The same arguments, scores and draws: ``keep_document`` draws from numpy's global random state
    (seeded once, when the filter is created), as NeMo does, also in pyroid's batched stages.
    """

    def __init__(
        self,
        model_path: str | None = None,
        label: str = "__label__hq",
        alpha: float = 3,
        seed: int = 42,
    ) -> None:
        if model_path is None:
            msg = (
                "Must provide a valid path to a FastText model to compute document scores "
                "with this filter"
            )
            raise ValueError(msg)
        self._model_path = model_path
        self._label = label
        self._alpha = alpha
        self._seed = np.random.seed(seed)
        self._name = "fasttext_quality_filter"

    def model_check_or_download(self) -> None:
        """Raise FileNotFoundError if the model file is missing, as NeMo does."""
        import os

        if not os.path.exists(self._model_path):
            msg = f"Model file {self._model_path} not found"
            raise FileNotFoundError(msg)

    def load_model(self) -> None:
        """Load the model with :func:`pyroid.fasttext.load_model`."""
        import pyroid.fasttext

        self._fasttext_quality_filter_model = pyroid.fasttext.load_model(self._model_path)

    def score_document(self, text: str) -> float:
        """The probability of ``label`` (or 1 minus the top label's), as NeMo computes it."""
        model = self._fasttext_quality_filter_model
        text = text.replace("\n", " ").replace("__label__", " ")
        label, score = model.predict([text])
        document_score: float = score[0][0].item()
        if label[0][0] != self._label:
            document_score = 1 - document_score
        return document_score

    def keep_document(self, score: float) -> bool:
        """Draw from numpy's global random state and compare, as NeMo does."""
        return bool(np.random.pareto(self._alpha) > 1 - score)


class FastTextLangId(_LangIdBase):
    """NeMo's ``FastTextLangId``, with the model loaded by :mod:`pyroid.fasttext`.

    The same arguments and the same score strings (``str([score, lang])``) as NeMo's.
    """

    def __init__(
        self, model_path: str | None = None, min_langid_score: float = 0.3, lang: str | None = None
    ) -> None:
        if model_path is None:
            msg = (
                "Must provide a valid path to a FastText model to identify languages "
                "with this filter"
            )
            raise ValueError(msg)
        self._model_path = model_path
        self._lang_code = lang or None
        self._cutoff = min_langid_score
        self._name = "lang_id"

    def model_check_or_download(self) -> None:
        """Raise FileNotFoundError if the model file is missing, as NeMo does."""
        import os

        if not os.path.exists(self._model_path):
            msg = f"Model file {self._model_path} not found"
            raise FileNotFoundError(msg)

    def load_model(self) -> None:
        """Load the model with :func:`pyroid.fasttext.load_model`."""
        import pyroid.fasttext

        self._fasttext_langid_model = pyroid.fasttext.load_model(self._model_path)

    def score_document(self, text: str) -> str:
        """NeMo's score string ``str([score, lang])`` for one document."""
        model = self._fasttext_langid_model
        pp = text.strip().replace("\n", " ")
        label, score = model.predict([pp], k=1)
        score = score[0][0].item()
        lang_code = label[0][0].removeprefix("__label__")
        return str([score, lang_code])

    def keep_document(self, score: float | str) -> bool:
        """NeMo's decision for a score string."""
        if isinstance(score, str):
            # NeMo parses its score strings with eval(); literal_eval gives the same values for
            # every string score_document makes and never runs code from a score column.
            score_lang = ast.literal_eval(score)
            value: Any = score_lang[0]
            lang = score_lang[1].casefold()
        else:
            msg = "score must be a string convertible to list"
            raise TypeError(msg)
        if self._lang_code:
            return bool(value >= self._cutoff and self._lang_matches(lang))
        return bool(value >= self._cutoff)

    def _lang_matches(self, lang: str) -> bool:
        """NeMo's language test for a casefolded label (``lang`` given)."""
        lang_filter = str(self._lang_code).casefold()
        if "_" in lang_filter:
            return bool(lang == lang_filter)
        return bool(lang.split("_", maxsplit=1)[0] == lang_filter)


def _fasttext_model(f: Any, attr: str) -> Any:
    import pyroid.fasttext

    model = getattr(f, attr, None)
    return model if isinstance(model, pyroid.fasttext.Model) else None


def _langid_batch(
    f: FastTextLangId, series: pd.Series, threads: int | None
) -> tuple[pd.Series, np.ndarray | None] | None:
    model = _fasttext_model(f, "_fasttext_langid_model")
    texts = _texts(series) if model is not None and len(series) else None
    if texts is None:
        return None
    try:
        probs, langs = _k.fasttext_lang_id(model, texts, threads=threads)
        strings = _k.lang_id_strings(probs, langs)
    except _FALLBACK_ERRORS:
        return None
    if not (_is_number(f._cutoff) and (f._lang_code is None or type(f._lang_code) is str)):
        keep = None
    else:
        # keep_document parses back exactly these floats and strings (repr round-trips): decide once
        # per distinct language, then compare the scores as arrays.
        keep = probs >= f._cutoff
        if f._lang_code:
            codes, uniq = pd.factorize(pd.Series(langs, dtype=object))
            matches = np.array([f._lang_matches(str(lang).casefold()) for lang in uniq], dtype=bool)
            keep &= matches[codes]
    return _like_apply(series, strings), keep


def _quality_batch(
    f: FastTextQualityFilter, series: pd.Series, threads: int | None
) -> pd.Series | None:
    model = _fasttext_model(f, "_fasttext_quality_filter_model")
    texts = _texts(series) if model is not None and len(series) else None
    if texts is None or type(f._label) is not str:
        return None
    try:
        scores = _k.fasttext_quality(model, texts, f._label, threads=threads)
    except _FALLBACK_ERRORS:
        return None
    return _like_apply(series, scores)


def _quality_keep(f: FastTextQualityFilter, scores: pd.Series) -> np.ndarray | None:
    if len(scores) == 0 or scores.hasnans or scores.dtype.kind != "f":
        return None
    values = scores.to_numpy(dtype=np.float64)
    # np.random.pareto(a, size=n) draws the same numbers as n calls of np.random.pareto(a).
    return np.asarray(np.random.pareto(f._alpha, size=len(values)) > 1 - values, dtype=bool)


# --- stages ---------------------------------------------------------------------------------------


def _score_and_keep(
    f: Any, series: pd.Series, threads: int | None, need_keep: bool
) -> tuple[pd.Series, np.ndarray | None] | None:
    """Scores (and, if possible, the keep mask) for filter ``f`` over ``series``, or None."""
    if type(f) is FastTextLangId:
        return _langid_batch(f, series, threads)
    if type(f) is FastTextQualityFilter:
        scores = _quality_batch(f, series, threads)
        if scores is None:
            return None
        return scores, (_quality_keep(f, scores) if need_keep else None)
    scores = _score_batch(f, series, threads)
    if scores is None:
        return None
    return scores, (_keep_batch(f, scores) if need_keep else None)


def _check_threads(threads: object) -> None:
    if threads is not None and (type(threads) is not int or threads < 1):
        msg = f"threads must be None or a positive int, not {threads!r}"
        raise ValueError(msg)


@dataclass
class ScoreFilter(_nc_sf.ScoreFilter):
    """NeMo's ``ScoreFilter``, scoring and filtering each batch in one pyroid call.

    The same arguments as NeMo's, plus ``threads`` (keyword; default 1).
    """

    threads: int | None = dataclasses.field(default=1, kw_only=True)

    def __post_init__(self) -> None:
        _check_threads(self.threads)
        super().__post_init__()

    def compute_filter_mask(
        self,
        df: pd.DataFrame,
        filter_obj: Any,
        text_field: str,
        score_field: str | None,
        invert: bool,
    ) -> pd.Series:
        """Score ``df[text_field]`` in one pyroid call and return NeMo's keep mask."""
        done = _score_and_keep(filter_obj, df[text_field], self.threads, need_keep=True)
        if done is None:
            return super().compute_filter_mask(df, filter_obj, text_field, score_field, invert)
        scores, keep = done
        if score_field is not None:
            df[score_field] = scores
        bool_mask = (
            scores.apply(filter_obj.keep_document)
            if keep is None
            else pd.Series(keep, index=df.index)
        )
        if invert:
            bool_mask = ~bool_mask
        return bool_mask


@dataclass
class Score(_nc_sf.Score):
    """NeMo's ``Score``, scoring each batch in one pyroid call.

    The same arguments as NeMo's, plus ``threads`` (keyword; default 1).
    """

    threads: int | None = dataclasses.field(default=1, kw_only=True)

    def __post_init__(self) -> None:
        _check_threads(self.threads)
        super().__post_init__()

    def process(self, batch: Any) -> Any:
        """Score the batch, one pyroid call per filter."""
        df = batch.to_pandas()
        if df.empty:
            return super().process(batch)
        for score_fn_i, text_field_i, score_field_i in zip(
            self.score_fn, self.text_field, self.score_field, strict=True
        ):
            done = _score_and_keep(score_fn_i, df[text_field_i], self.threads, need_keep=False)
            if done is not None:
                df[score_field_i] = done[0]
                continue
            inner = (
                score_fn_i.score_document
                if isinstance(score_fn_i, _nc_doc_filter.DocumentFilter)
                else score_fn_i
            )
            df[score_field_i] = df[text_field_i].apply(inner)
        return type(batch)(
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


@dataclass
class Filter(_nc_sf.Filter):
    """NeMo's ``Filter``, deciding each batch with array comparisons.

    The same arguments as NeMo's, plus ``threads`` (keyword; default 1; unused, for symmetry).
    """

    threads: int | None = dataclasses.field(default=1, kw_only=True)

    def __post_init__(self) -> None:
        _check_threads(self.threads)
        super().__post_init__()

    def compute_filter_mask(
        self, df: pd.DataFrame, filter_fn: Any, filter_field: str, invert: bool
    ) -> pd.Series:
        """NeMo's keep mask for the scores in ``df[filter_field]``, as array comparisons."""
        scores = df[filter_field]
        if type(filter_fn) is FastTextQualityFilter:
            keep = _quality_keep(filter_fn, scores)
        elif type(filter_fn) in _FILTERS:
            keep = _keep_batch(filter_fn, scores)
        else:
            keep = None
        if keep is None:
            return super().compute_filter_mask(df, filter_fn, filter_field, invert)
        bool_mask = pd.Series(keep, index=df.index)
        if invert:
            bool_mask = ~bool_mask
        return bool_mask


# --- modifiers -----------------------------------------------------------------------------


# NeMo's own classes, captured now: pyroid.patch("nemo_curator") later puts pyroid's in their place.
_NC_BOILERPLATE = _nc_ms.BoilerPlateStringModifier
_NC_MARKDOWN = _nc_ms.MarkdownRemover
_NC_NEWLINE = _nc_ms.NewlineNormalizer
_NC_QUOTATION = _nc_ms.QuotationRemover
_NC_URL = _nc_ms.UrlRemover
_NC_LINES = _nc_ms.LineRemover
_NC_SLICER = _nc_ms.Slicer
_NC_LABEL = _nc_ftl.FastTextLabelModifier


def _modify_plan(m: Any) -> Callable[[Any, int | None], Any] | None:
    """Pyroid's transform for modifier ``m`` (from its current attributes), or None."""
    t = type(m)
    if t in (_NC_BOILERPLATE, BoilerPlateStringModifier):
        if m._paragraphs is not None:
            _fallback(t.__name__, "precomputed paragraphs")
            return None
        remove = bool(m._remove_if_at_top_or_bottom)
        return lambda x, th: _k.remove_boilerplate_strings(
            x, remove_if_at_top_or_bottom=remove, threads=th
        )
    if t in (_NC_MARKDOWN, MarkdownRemover):
        return lambda x, th: _k.remove_markdown(x, threads=th)
    if t in (_NC_NEWLINE, NewlineNormalizer):
        return lambda x, th: _k.normalize_newlines(x, threads=th)
    if t in (_NC_QUOTATION, QuotationRemover):
        return lambda x, th: _k.remove_quotation(x, threads=th)
    if t in (_NC_URL, UrlRemover):
        return lambda x, th: _k.remove_urls(x, threads=th)
    if t in (_NC_LINES, LineRemover):
        patterns = m._patterns
        if not isinstance(patterns, (list, tuple, set, frozenset)) or not all(
            type(p) is str for p in patterns
        ):
            _fallback(t.__name__, "non-str patterns")
            return None
        plist = list(patterns)
        return lambda x, th: _k.remove_lines(x, plist, threads=th)
    if t in (_NC_SLICER, Slicer):
        args = (m._left, m._right, m._include_left, m._include_right, m._strip)
        return lambda x, th: _k.slice_text(x, *args, threads=th)
    if t in (_NC_LABEL, FastTextLabelModifier):
        label = m.label
        return lambda x, th: _k.fasttext_label(x, label, threads=th)
    if t is UnicodeReformatter or (
        _UnicodeBase is not _nc_doc_modifier.DocumentModifier and t is _UnicodeBase
    ):
        config = m.config
        return lambda x, th: pyroid.ftfy.fix_text_batch(x, config, threads=th)
    return None


def _modify_batch(m: Any, series: pd.Series, threads: int | None) -> pd.Series | None:
    fn = _modify_plan(m)
    if fn is None or len(series) == 0:
        return None
    texts = _texts(series)
    if texts is None:
        return None
    try:
        out = fn(texts, threads)
    except _FALLBACK_ERRORS:
        return None
    if not isinstance(out, list):
        import pyarrow as pa

        out = pa.chunked_array(out)
    return _like_apply(series, out)


@dataclass
class Modify(_nc_modifier.Modify):
    """NeMo's ``Modify``, modifying each batch in one pyroid call.

    The same arguments as NeMo's, plus ``threads`` (keyword; default 1). Modifiers with several
    input fields run NeMo's code.
    """

    threads: int | None = dataclasses.field(default=1, kw_only=True)

    def __post_init__(self) -> None:
        _check_threads(self.threads)
        super().__post_init__()

    def process(self, batch: Any) -> Any:
        """Modify the batch, one pyroid call per modifier."""
        df = batch.to_pandas()
        for modifier_fn_i, input_fields_i, output_field_i in zip(
            self.modifier_fn, self._input_fields, self._output_fields, strict=True
        ):
            if len(input_fields_i) == 1:
                src = input_fields_i[0]
                out = _modify_batch(modifier_fn_i, df[src], self.threads)
                if out is not None:
                    df[output_field_i] = out
                    continue
                inner = (
                    modifier_fn_i.modify_document
                    if isinstance(modifier_fn_i, _nc_doc_modifier.DocumentModifier)
                    else modifier_fn_i
                )
                df[output_field_i] = df[src].apply(inner)
            else:
                inner = (
                    modifier_fn_i.modify_document
                    if isinstance(modifier_fn_i, _nc_doc_modifier.DocumentModifier)
                    else modifier_fn_i
                )
                df[output_field_i] = [inner(**rec) for rec in df[input_fields_i].to_dict("records")]
        return type(batch)(
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


class _OneModify:
    """``modify_document`` through pyroid for one document, NeMo's code otherwise."""

    def modify_document(self, text: str) -> Any:
        fn = _modify_plan(self)
        if fn is not None and type(text) is str:
            try:
                return fn(text, 1)
            except _FALLBACK_ERRORS:
                pass
        return super().modify_document(text)  # type: ignore[misc]


class BoilerPlateStringModifier(_OneModify, _nc_ms.BoilerPlateStringModifier):
    """NeMo's ``BoilerPlateStringModifier`` via :func:`pyroid.nemo.remove_boilerplate_strings`."""


class MarkdownRemover(_OneModify, _nc_ms.MarkdownRemover):
    """NeMo's ``MarkdownRemover`` through :func:`pyroid.nemo.remove_markdown`."""


class NewlineNormalizer(_OneModify, _nc_ms.NewlineNormalizer):
    """NeMo's ``NewlineNormalizer`` through :func:`pyroid.nemo.normalize_newlines`."""


class QuotationRemover(_OneModify, _nc_ms.QuotationRemover):
    """NeMo's ``QuotationRemover`` through :func:`pyroid.nemo.remove_quotation`."""


class UrlRemover(_OneModify, _nc_ms.UrlRemover):
    """NeMo's ``UrlRemover`` through :func:`pyroid.nemo.remove_urls`."""


class LineRemover(_OneModify, _nc_ms.LineRemover):
    """NeMo's ``LineRemover`` through :func:`pyroid.nemo.remove_lines`."""


class Slicer(_OneModify, _nc_ms.Slicer):
    """NeMo's ``Slicer`` through :func:`pyroid.nemo.slice_text`."""


class FastTextLabelModifier(_OneModify, _nc_ftl.FastTextLabelModifier):
    """NeMo's ``FastTextLabelModifier`` through :func:`pyroid.nemo.fasttext_label`."""


try:
    from nemo_curator.stages.text.modifiers.unicode import unicode_reformatter as _nc_ur

    _UnicodeBase: Any = _nc_ur.UnicodeReformatter
except ImportError:  # ftfy is not installed: NeMo's module imports it at the top
    _UnicodeBase = _nc_doc_modifier.DocumentModifier


class UnicodeReformatter(_UnicodeBase):
    """NeMo's ``UnicodeReformatter`` through :mod:`pyroid.ftfy` (ftfy itself is not needed).

    The same arguments and defaults as NeMo's, which differ from ftfy's: no Unicode normalization,
    and ligatures, character widths, curly quotes and line breaks are left alone.
    """

    def __init__(
        self,
        config: Any = None,
        unescape_html: str | bool = "auto",
        remove_terminal_escapes: bool = True,
        fix_encoding: bool = True,
        restore_byte_a0: bool = True,
        replace_lossy_sequences: bool = True,
        decode_inconsistent_utf8: bool = True,
        fix_c1_controls: bool = True,
        fix_latin_ligatures: bool = False,
        fix_character_width: bool = False,
        uncurl_quotes: bool = False,
        fix_line_breaks: bool = False,
        fix_surrogates: bool = True,
        remove_control_chars: bool = True,
        normalization: Literal["NFC", "NFD", "NFKC", "NFKD"] | None = None,
        max_decode_length: int = 1000000,
        explain: bool = True,
    ) -> None:
        if config is None:
            self.config = pyroid.ftfy.TextFixerConfig(
                unescape_html=unescape_html,
                remove_terminal_escapes=remove_terminal_escapes,
                fix_encoding=fix_encoding,
                restore_byte_a0=restore_byte_a0,
                replace_lossy_sequences=replace_lossy_sequences,
                decode_inconsistent_utf8=decode_inconsistent_utf8,
                fix_c1_controls=fix_c1_controls,
                fix_latin_ligatures=fix_latin_ligatures,
                fix_character_width=fix_character_width,
                uncurl_quotes=uncurl_quotes,
                fix_line_breaks=fix_line_breaks,
                fix_surrogates=fix_surrogates,
                remove_control_chars=remove_control_chars,
                normalization=normalization,
                max_decode_length=max_decode_length,
                explain=explain,
            )
        else:
            self.config = config
        _nc_doc_modifier.DocumentModifier.__init__(self)

    def modify_document(self, text: str) -> str:
        """``pyroid.ftfy.fix_text(text, config=self.config)``."""
        return pyroid.ftfy.fix_text(text, config=self.config)


# --- heuristic filters ---------------------------------------------------------------------


class NonAlphaNumericFilter(_OneDoc, _nc_hs.NonAlphaNumericFilter):
    """NeMo's ``NonAlphaNumericFilter`` through :func:`pyroid.nemo.non_alpha_numeric`."""


class SymbolsToWordsFilter(_OneDoc, _nc_hs.SymbolsToWordsFilter):
    """NeMo's ``SymbolsToWordsFilter`` through :func:`pyroid.nemo.symbols_to_words`."""


class NumbersFilter(_OneDoc, _nc_hs.NumbersFilter):
    """NeMo's ``NumbersFilter`` through :func:`pyroid.nemo.numbers`."""


class UrlsFilter(_OneDoc, _nc_hs.UrlsFilter):
    """NeMo's ``UrlsFilter`` through :func:`pyroid.nemo.urls` (NeMo's code for a custom regex)."""


class BulletsFilter(_OneDoc, _nc_hs.BulletsFilter):
    """NeMo's ``BulletsFilter`` through :func:`pyroid.nemo.bullets`."""


class WhiteSpaceFilter(_OneDoc, _nc_hs.WhiteSpaceFilter):
    """NeMo's ``WhiteSpaceFilter`` through :func:`pyroid.nemo.white_space`."""


class ParenthesesFilter(_OneDoc, _nc_hs.ParenthesesFilter):
    """NeMo's ``ParenthesesFilter`` through :func:`pyroid.nemo.parentheses`."""


class LongWordFilter(_OneDoc, _nc_hs.LongWordFilter):
    """NeMo's ``LongWordFilter`` through :func:`pyroid.nemo.long_word`."""


class WordCountFilter(_OneDoc, _nc_hs.WordCountFilter):
    """NeMo's ``WordCountFilter`` through :func:`pyroid.nemo.word_count`."""


class BoilerPlateStringFilter(_OneDoc, _nc_hs.BoilerPlateStringFilter):
    """NeMo's ``BoilerPlateStringFilter`` through :func:`pyroid.nemo.boilerplate_string`."""


class MeanWordLengthFilter(_OneDoc, _nc_hs.MeanWordLengthFilter):
    """NeMo's ``MeanWordLengthFilter`` through :func:`pyroid.nemo.mean_word_length`."""


class PunctuationFilter(_OneDoc, _nc_hs.PunctuationFilter):
    """NeMo's ``PunctuationFilter`` through :func:`pyroid.nemo.punctuation`."""


class EllipsisFilter(_OneDoc, _nc_hs.EllipsisFilter):
    """NeMo's ``EllipsisFilter`` through :func:`pyroid.nemo.ellipsis`."""


class CommonEnglishWordsFilter(_OneDoc, _nc_hs.CommonEnglishWordsFilter):
    """NeMo's ``CommonEnglishWordsFilter`` through :func:`pyroid.nemo.common_english_words`."""


class WordsWithoutAlphabetsFilter(_OneDoc, _nc_hs.WordsWithoutAlphabetsFilter):
    """NeMo's ``WordsWithoutAlphabetsFilter`` via :func:`pyroid.nemo.words_without_alphabets`."""


class PornographicUrlsFilter(_OneDoc, _nc_hs.PornographicUrlsFilter):
    """NeMo's ``PornographicUrlsFilter`` through :func:`pyroid.nemo.pornographic_urls`."""


class SubstringFilter(_OneDoc, _nc_hs.SubstringFilter):
    """NeMo's ``SubstringFilter`` through :func:`pyroid.nemo.substring`."""


class RepeatedLinesFilter(_OneDoc, _nc_hr.RepeatedLinesFilter):
    """NeMo's ``RepeatedLinesFilter`` through :func:`pyroid.nemo.repeated_lines`."""


class RepeatedParagraphsFilter(_OneDoc, _nc_hr.RepeatedParagraphsFilter):
    """NeMo's ``RepeatedParagraphsFilter`` through :func:`pyroid.nemo.repeated_paragraphs`."""


class RepeatedLinesByCharFilter(_OneDoc, _nc_hr.RepeatedLinesByCharFilter):
    """NeMo's ``RepeatedLinesByCharFilter`` through :func:`pyroid.nemo.repeated_lines_by_char`."""


class RepeatedParagraphsByCharFilter(_OneDoc, _nc_hr.RepeatedParagraphsByCharFilter):
    """NeMo's ``RepeatedParagraphsByCharFilter``, via :mod:`pyroid.nemo`."""


class RepeatingTopNGramsFilter(_OneDoc, _nc_hr.RepeatingTopNGramsFilter):
    """NeMo's ``RepeatingTopNGramsFilter`` through :func:`pyroid.nemo.repeating_top_ngrams`."""


class RepeatingDuplicateNGramsFilter(_OneDoc, _nc_hr.RepeatingDuplicateNGramsFilter):
    """NeMo's ``RepeatingDuplicateNGramsFilter``, via :mod:`pyroid.nemo`."""


# pyroid's subclasses use the same plans as NeMo's classes (exact types only: a further subclass
# may override score_document, so it runs its own code).
for _ours in (
    NonAlphaNumericFilter, SymbolsToWordsFilter, NumbersFilter, UrlsFilter, BulletsFilter,
    WhiteSpaceFilter, ParenthesesFilter, LongWordFilter, WordCountFilter, BoilerPlateStringFilter,
    MeanWordLengthFilter, PunctuationFilter, EllipsisFilter, CommonEnglishWordsFilter,
    WordsWithoutAlphabetsFilter, PornographicUrlsFilter, SubstringFilter, RepeatedLinesFilter,
    RepeatedParagraphsFilter, RepeatedLinesByCharFilter, RepeatedParagraphsByCharFilter,
    RepeatingTopNGramsFilter, RepeatingDuplicateNGramsFilter,
):  # fmt: skip
    _FILTERS[_ours] = _FILTERS[_ours.__mro__[2]]
del _ours

# NeMo registers every ProcessingStage subclass by class name; keep its names pointing at its own.
for _stock in (_nc_sf.ScoreFilter, _nc_sf.Score, _nc_sf.Filter, _nc_modifier.Modify):
    _nc_base._STAGE_REGISTRY[_stock.__name__] = _stock
del _stock
