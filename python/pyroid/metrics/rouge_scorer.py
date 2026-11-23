# Derived from rouge-score 0.1.2 (Apache-2.0), © The rouge_score Authors; modified.
"""A drop-in, Rust-backed replacement for ``rouge_score.rouge_scorer`` (rouge-score 0.1.2).

``RougeScorer`` has rouge-score's signature and returns the same
``dict[str, Score]``, float for float. Tokenizing, Porter stemming (a port of
nltk 3.10.4's) and scoring run in Rust. What needs the interpreter stays in
Python, exactly where rouge-score runs it: a custom ``tokenizer`` is called
from Python (its tokens are scored in Rust), and ``split_summaries=True``
splits rougeLsum sentences with :func:`pyroid.nltk.sent_tokenize`, the Rust
port of ``nltk.sent_tokenize`` (it reads NLTK's ``punkt_tab`` model). Tokens
that are not lists of str (from a custom tokenizer) are scored by the real
rouge-score, with a one-time :class:`pyroid.FallbackWarning`.

``RougeScorer.score_batch`` (a pyroid extension) scores many pairs at once on
pyroid's thread pool.
"""

from __future__ import annotations

import logging
import re
import sys
import warnings
from typing import Any

from pyroid import _versions
from pyroid._pyroid import FallbackWarning, InputError, PyroidError
from pyroid._pyroid import metrics as _native
from pyroid.metrics import scoring, tokenizers
from pyroid.metrics.tokenize import _NLTK, _ensure_str

__all__ = ["ROUGE_SCORE_VERSION", "RougeScorer"]

#: The rouge-score release whose behaviour this module reproduces.
ROUGE_SCORE_VERSION = "0.1.2"

_versions.check(__name__, "rouge_score", f"rouge-score=={ROUGE_SCORE_VERSION}")
_versions.check(__name__, "nltk", _NLTK)  # the Porter stemmer

_log = logging.getLogger(__name__)
_ROUGE_N = re.compile(r"rouge[0-9]$")
_DefaultTokenizer = tokenizers.DefaultTokenizer
_score_native = _native.rouge_score
# Job kinds of _native.rouge_score_tokens (n >= 1 is rouge-n).
_KIND_L = 0
_KIND_LSUM = -1

_WARNED: set[str] = set()


def _caller_stacklevel() -> int:
    """The ``stacklevel`` that points a warning at the first frame outside this module."""
    frame = sys._getframe(1)
    level = 1
    while frame.f_back is not None and frame.f_globals.get("__name__") == __name__:
        frame = frame.f_back
        level += 1
    return level


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(message, FallbackWarning, stacklevel=_caller_stacklevel())


def _rouge_score(reason: str) -> Any:
    """The real ``rouge_score.rouge_scorer`` to delegate to (warns once), or PyroidError."""
    try:
        from rouge_score import rouge_scorer as real
    except ImportError:
        raise PyroidError(
            f"pyroid.metrics.rouge_scorer: {reason} are not supported by pyroid's Rust port, "
            f"and rouge-score is not installed to fall back to "
            f"(pip install rouge-score=={ROUGE_SCORE_VERSION})"
        ) from None
    _warn_once(
        reason,
        f"pyroid.metrics.rouge_scorer: {reason} are not supported by the Rust port; "
        "scoring them with rouge-score (slow path). This warning is shown once.",
    )
    return real


def _sent_tokenize(text: Any) -> list[str]:
    """``nltk.sent_tokenize(text)`` for ``split_summaries=True``: pyroid.nltk's Punkt port.

    The model is NLTK's English ``punkt_tab`` (``LookupError`` if it is not installed). Text
    that is not a str goes to NLTK itself, which raises for it as rouge-score would.
    """
    if type(text) is str:
        from pyroid.nltk import sent_tokenize

        return sent_tokenize(text)  # type: ignore[no-any-return]
    try:
        import nltk
    except ImportError:
        raise PyroidError(
            "pyroid.metrics.rouge_scorer: split_summaries=True got a text that is not a str, "
            "which only nltk.sent_tokenize handles, and nltk is not installed"
        ) from None
    return nltk.sent_tokenize(text)  # type: ignore[no-any-return]


_TOKENS_REASON = "tokens that are not lists of str (from a custom tokenizer)"


def _from_tokens(kind: int, target: Any, prediction: Any) -> scoring.Score:
    """rouge-score's score of one type from (lists of) token lists."""
    score = _native.rouge_score_tokens([(kind, target, prediction)], threads=1)[0]
    if score is not None:
        return score  # type: ignore[no-any-return]
    real = _rouge_score(_TOKENS_REASON)
    if kind == _KIND_L:
        s = real._score_lcs(target, prediction)
    elif kind == _KIND_LSUM:
        s = real._summary_level_lcs(target, prediction)
    else:
        s = real._score_ngrams(
            real._create_ngrams(target, kind), real._create_ngrams(prediction, kind)
        )
    return scoring.Score(*s)


def _valid_types(types: Any) -> bool:
    """Are ``types`` an exact list/tuple of exact strs (the native paths)?

    If so, raise rouge-score's ``ValueError`` for the first invalid one.
    """
    if (type(types) is not list and type(types) is not tuple) or any(
        type(t) is not str for t in types
    ):
        return False
    for t in types:
        if t in ("rougeL", "rougeLsum"):
            continue
        if not _ROUGE_N.match(t):
            raise ValueError("Invalid rouge type: %s" % t)  # noqa: UP031
        if int(t[5:]) <= 0:
            raise ValueError("rougen requires positive n: %s" % t)  # noqa: UP031
    return True


def _argmax(values: list[Any]) -> int:
    """``np.argmax(values)`` for a list of numbers: the first maximum."""
    if not values:
        raise ValueError("attempt to get argmax of an empty sequence")
    best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best


def _texts(texts: Any, name: str) -> Any:
    """A batch argument as a Python sequence (Arrow arrays become a list, None for nulls)."""
    if isinstance(texts, (list, tuple)):
        return texts
    return _native.batch_texts(texts, name)


def _first_none(texts: Any) -> int | None:
    """The index of the first item that is ``None`` (by identity: items may be anything)."""
    for i, t in enumerate(texts):
        if t is None:
            return i
    return None


def _check_no_none(texts: Any, name: str) -> None:
    """Raise :class:`pyroid.InputError` for a list or tuple holding ``None``."""
    if isinstance(texts, (list, tuple)):
        i = _first_none(texts)
        if i is not None:
            err = InputError(f"expected str at index {i}, got NoneType")
            err.add_note(f"while processing {name!r}")
            raise err


def _all_str(texts: Any) -> bool:
    """False for a list/tuple holding anything but exact strs (Arrow: True)."""
    if isinstance(texts, (list, tuple)):
        return all(type(t) is str for t in texts)
    return True


def _all_str_or_null(texts: Any) -> bool:
    """:func:`_all_str` for the result of :func:`_texts`, where ``None`` is an Arrow null."""
    return all(t is None or type(t) is str for t in texts)


def _without_nulls(targets: Any, predictions: Any) -> tuple[Any, Any, list[int] | None]:
    """The pairs with no ``None`` (an Arrow null) on either side, and their indices.

    The indices are None when every pair is kept.
    """
    if _first_none(targets) is None and _first_none(predictions) is None:
        return targets, predictions, None
    keep = [
        i
        for i, (t, p) in enumerate(zip(targets, predictions, strict=True))
        if t is not None and p is not None
    ]
    return [targets[i] for i in keep], [predictions[i] for i in keep], keep


def _with_nulls(results: list[Any], keep: list[int] | None, n: int) -> list[Any]:
    """``results`` of the kept pairs back in place, ``None`` for the others."""
    if keep is None:
        return results
    out: list[Any] = [None] * n
    for i, r in zip(keep, results, strict=True):
        out[i] = r
    return out


class RougeScorer(scoring.BaseScorer):
    """Calculate rouge scores between two blobs of text (rouge-score's ``RougeScorer``).

    Sample usage:
        ```python
        scorer = RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        scores = scorer.score("The quick brown fox jumps over the lazy dog",
                              "The quick brown dog jumps on the log.")
        ```
    """

    def __init__(
        self,
        rouge_types: Any,
        use_stemmer: bool = False,
        split_summaries: bool = False,
        tokenizer: Any = None,
    ) -> None:
        """Initializes a new RougeScorer.

        Valid rouge types that can be computed are:

        - ``rouge1`` .. ``rouge9``: n-gram based scoring.
        - ``rougeL``: longest common subsequence based scoring.
        - ``rougeLsum``: summary-level LCS over sentences (lines, or Punkt
          sentences with ``split_summaries``).

        Types are checked when scoring, as in rouge-score.

        Args:
            rouge_types: A list of rouge types to calculate.
            use_stemmer: Bool indicating whether Porter stemmer should be used to
                strip word suffixes to improve matching. This arg is used in the
                DefaultTokenizer, but other tokenizers might or might not choose to
                use this.
            split_summaries: whether to add newlines between sentences for rougeLsum
                (sentences are split with :func:`pyroid.nltk.sent_tokenize`).
            tokenizer: Tokenizer object which has a tokenize() method. A falsy
                value means the default tokenizer.
        """
        self.rouge_types = rouge_types
        if tokenizer:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = tokenizers.DefaultTokenizer(use_stemmer)
            _log.info("Using default tokenizer.")

        self._split_summaries = split_summaries

    def score_multi(self, targets: Any, prediction: Any) -> dict[Any, scoring.Score]:
        """Calculates rouge scores between targets and prediction.

        The target with the maximum f-measure is used for the final score for
        each score type (the first one on ties).

        Args:
            targets: list of texts containing the targets
            prediction: Text containing the predicted text.

        Returns:
            A dict mapping each rouge type to a Score object.

        Raises:
            ValueError: If an invalid rouge type is encountered, or ``targets``
                is empty.
        """
        score_dicts = [self.score(t, prediction) for t in targets]
        max_score = {}
        for k in self.rouge_types:
            index = _argmax([s[k].fmeasure for s in score_dicts])
            max_score[k] = score_dicts[index][k]

        return max_score

    def score(self, target: Any, prediction: Any) -> dict[Any, scoring.Score]:
        """Calculates rouge scores between the target and prediction.

        Args:
            target: Text containing the target (ground truth) text.
            prediction: Text containing the predicted text.

        Returns:
            A dict mapping each rouge type to a Score object, in
            ``rouge_types`` order.

        Raises:
            ValueError: If an invalid rouge type is encountered
                (``"Invalid rouge type: ..."``, or ``"rougen requires positive
                n: ..."`` for ``rouge0``).
            UnicodeEncodeError: ``target`` or ``prediction`` holds a lone
                surrogate (default tokenizer).
        """
        tokenizer = self._tokenizer
        if type(tokenizer) is _DefaultTokenizer and type(target) is str and type(prediction) is str:
            if not self._split_summaries:
                result = _score_native(self.rouge_types, target, prediction, tokenizer._use_stemmer)
            else:
                result = self._score_split(target, prediction, tokenizer._use_stemmer)
            if result is not None:
                return result  # type: ignore[no-any-return]
        return self._score_python(target, prediction)

    def score_batch(
        self, targets: Any, predictions: Any, *, threads: int | None = None
    ) -> list[dict[Any, scoring.Score] | None]:
        """Score many pairs at once, on pyroid's thread pool (not part of rouge-score).

        Element ``i`` equals ``self.score(targets[i], predictions[i])``. With the
        default tokenizer and a list/tuple of rouge type strs, every pair is
        tokenized and scored in Rust in parallel (with ``split_summaries=True``,
        sentences are split in parallel too). Otherwise (a custom tokenizer,
        list items that are not ``str``) the pairs are scored one by one with
        :meth:`score`.

        Args:
            targets: The targets: a list or tuple of str, or an Arrow string
                array/column (``string``, ``large_string`` or ``string_view``;
                pyarrow, polars, ``pyroid.Column``, ...). Not a single str:
                use :meth:`score` for one pair.
            predictions: The predictions, as for ``targets``, of the same length.
            threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at
                most ``n`` tasks at once; ``1`` runs on the calling thread.

        Returns:
            A list (for any input kind) with one ``dict[str, Score]`` per pair,
            in order, and ``None`` for a pair with an Arrow null on either side.

        Raises:
            ValueError: ``targets`` and ``predictions`` differ in length, or a
                rouge type is invalid (raised before any pair is scored, even
                for an empty batch).
            pyroid.InputError: ``targets`` or ``predictions`` is not a list,
                tuple or Arrow string array (a single str is rejected), a list
                or tuple holds ``None`` (the message gives its index), or
                ``threads`` is not an int.
            UnicodeEncodeError: A str holds a lone surrogate.
            UnicodeDecodeError: An Arrow value is not valid UTF-8.
        """
        _check_no_none(targets, "targets")
        _check_no_none(predictions, "predictions")
        types = self.rouge_types
        tokenizer = self._tokenizer
        if _valid_types(types) and type(tokenizer) is _DefaultTokenizer:
            stem = tokenizer._use_stemmer
            result: Any = None
            if self._split_summaries and "rougeLsum" in types:
                targets = _texts(targets, "targets")
                predictions = _texts(predictions, "predictions")
                if _all_str_or_null(targets) and _all_str_or_null(predictions):
                    _check_lengths(targets, predictions)
                    ts, ps, keep = _without_nulls(targets, predictions)
                    from pyroid.nltk import sent_tokenize

                    target_sents = [
                        [s for s in d if len(s)] for d in sent_tokenize(list(ts), threads=threads)
                    ]
                    prediction_sents = [
                        [s for s in d if len(s)] for d in sent_tokenize(list(ps), threads=threads)
                    ]
                    result = _native.rouge_score_batch(
                        types, ts, ps, stem, target_sents, prediction_sents, threads=threads
                    )
                    if result is not None:
                        result = _with_nulls(result, keep, len(targets))
            elif _all_str(targets) and _all_str(predictions):
                result = _native.rouge_score_batch(
                    types, targets, predictions, stem, threads=threads
                )
            if result is not None:
                return result  # type: ignore[no-any-return]
        # Custom tokenizers, non-str items, unusual rouge_types: pair by pair.
        _native.rouge_score_tokens([], threads=threads)  # validates `threads`
        targets = _texts(targets, "targets")
        predictions = _texts(predictions, "predictions")
        _check_lengths(targets, predictions)
        ts, ps, keep = _without_nulls(targets, predictions)
        scores = [self.score(t, p) for t, p in zip(ts, ps, strict=True)]
        return _with_nulls(scores, keep, len(targets))

    def _sentences(self, text: Any) -> list[Any]:
        """rouge-score's ``get_sents``: rougeLsum's sentences of ``text``."""
        # Without split_summaries, sentences are assumed to be separated by newline.
        sents = _sent_tokenize(text) if self._split_summaries else _ensure_str(text).split("\n")
        return [x for x in sents if len(x)]

    def _score_split(self, target: str, prediction: str, use_stemmer: bool) -> Any:
        """``score`` with ``split_summaries=True``: nltk splits, Rust scores.

        nltk runs where rouge-score runs it, at the first rougeLsum (so a missing
        punkt_tab raises before a later invalid type, as in rouge-score).
        """
        types = self.rouge_types
        if type(types) is not list and type(types) is not tuple:
            return None
        sentences = None
        for t in types:
            if type(t) is not str:
                return None
            if t == "rougeLsum":
                if sentences is None:
                    sentences = (self._sentences(target), self._sentences(prediction))
            elif t != "rougeL" and not (_ROUGE_N.match(t) and int(t[5:]) > 0):
                break  # the native call raises rouge-score's ValueError for it
        if sentences is None:
            return _score_native(types, target, prediction, use_stemmer)
        return _score_native(types, target, prediction, use_stemmer, *sentences)

    def _score_python(self, target: Any, prediction: Any) -> dict[Any, scoring.Score]:
        """rouge-score's ``score``, line for line, with the scoring in Rust."""
        # Pre-compute target tokens and prediction tokens for use by different
        # types, except if only "rougeLsum" is requested.
        if len(self.rouge_types) == 1 and self.rouge_types[0] == "rougeLsum":
            target_tokens = None
            prediction_tokens = None
        else:
            target_tokens = self._tokenizer.tokenize(target)
            prediction_tokens = self._tokenizer.tokenize(prediction)
        result = {}

        for rouge_type in self.rouge_types:
            if rouge_type == "rougeL":
                # Rouge from longest common subsequences.
                scores = _from_tokens(_KIND_L, target_tokens, prediction_tokens)
            elif rouge_type == "rougeLsum":
                # Note: Does not support multi-line text.
                target_tokens_list = [self._tokenizer.tokenize(s) for s in self._sentences(target)]
                prediction_tokens_list = [
                    self._tokenizer.tokenize(s) for s in self._sentences(prediction)
                ]
                scores = _from_tokens(_KIND_LSUM, target_tokens_list, prediction_tokens_list)
            elif _ROUGE_N.match(_ensure_str(rouge_type)):
                # Rouge from n-grams.
                n = int(rouge_type[5:])
                if n <= 0:
                    raise ValueError("rougen requires positive n: %s" % rouge_type)  # noqa: UP031
                scores = _from_tokens(n, target_tokens, prediction_tokens)
            else:
                raise ValueError("Invalid rouge type: %s" % rouge_type)  # noqa: UP031
            result[rouge_type] = scores

        return result


def _check_lengths(targets: Any, predictions: Any) -> None:
    if len(targets) != len(predictions):
        raise ValueError(
            "score_batch: targets and predictions must have the same length, "
            f"got {len(targets)} and {len(predictions)}"
        )
