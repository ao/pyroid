# Derived from rouge-score 0.1.2 (Apache-2.0), © The rouge_score Authors; modified.
# Type stubs for pyroid.metrics.rouge_scorer. The module docstring is kept here
# on purpose: the API docs are built from this stub.
"""A drop-in, Rust-backed replacement for ``rouge_score.rouge_scorer`` (rouge-score 0.1.2).

``from pyroid.metrics import rouge_scorer, scoring`` replaces
``from rouge_score import rouge_scorer, scoring``. :class:`RougeScorer` takes
rouge-score's arguments and returns the same ``dict[str, Score]``: the same
floats bit for bit, ``int`` zeros where rouge-score has them, the same key
order, the same ``ValueError`` messages. ROUGE-N (``rouge1``..``rouge9``), ROUGE-L
and ROUGE-Lsum are supported, with or without nltk 3.10.4's Porter stemmer
(ported to Rust: nltk is not needed).

Tokenizing, stemming and scoring run in Rust. What needs the interpreter runs
in Python, exactly where rouge-score runs it:

- a custom ``tokenizer`` is called from Python (for the target, the
  prediction, and each rougeLsum sentence); its tokens are scored in Rust.
  Tokens that are not lists of ``str`` are scored by the real rouge-score,
  with a one-time :class:`pyroid.FallbackWarning`
  (:class:`pyroid.PyroidError` if rouge-score is not installed);
- ``split_summaries=True`` splits rougeLsum sentences with
  ``nltk.sent_tokenize`` (one-time ``FallbackWarning``; ``PyroidError`` if
  nltk is missing; nltk's ``LookupError`` for missing ``punkt_tab`` data
  propagates as in rouge-score).

:meth:`RougeScorer.score_batch` (a pyroid extension) scores many pairs at once
on pyroid's thread pool.

Example:
    ```python
    from pyroid.metrics import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scorer.score("The quick brown fox jumps over the lazy dog",
                 "The quick brown dog jumps on the log.")
    # {'rouge1': Score(precision=0.75, recall=0.666..., fmeasure=0.705...),
    #  'rougeL': Score(precision=0.625, recall=0.555..., fmeasure=0.588...)}
    scorer.score_batch(targets, predictions, threads=8)  # list of such dicts
    ```

Differences from rouge-score
----------------------------
- Lone surrogates (U+D800..U+DFFF) in a ``str`` target or prediction raise
  ``UnicodeEncodeError`` (with a note) when the default tokenizer is used;
  rouge-score treats them as separators. Remove them first, e.g.
  ``s.encode("utf-8", "replace").decode("utf-8")``. A custom tokenizer's
  tokens may hold them (those are scored by rouge-score).
- ``RougeScorer`` is pyroid's class, a :class:`pyroid.metrics.scoring.BaseScorer`;
  its scores are :class:`pyroid.metrics.scoring.Score` namedtuples, equal to
  rouge-score's but not instances of ``rouge_score.scoring.Score``.
- rouge-score's private helpers (``_score_lcs``, ``_lcs_table``,
  ``_summary_level_lcs``, ...) and ``lcs_ind`` are not provided.
- ``score_multi`` picks the first maximal f-measure as ``np.argmax`` does,
  without numpy (its error for no targets is numpy's
  ``ValueError("attempt to get argmax of an empty sequence")``).
- "Using default tokenizer." is logged at INFO level with :mod:`logging`
  (logger ``pyroid.metrics.rouge_scorer``), not with absl.
- nltk is imported only when ``split_summaries=True`` and a rougeLsum is
  scored; rouge-score always imports nltk, absl, numpy and six.
"""

from collections.abc import Iterable, Sequence
from typing import Any, Final, Protocol, type_check_only

from pyroid._typing import ArrowStrings
from pyroid.metrics import scoring

__all__ = ["ROUGE_SCORE_VERSION", "RougeScorer"]

ROUGE_SCORE_VERSION: Final[str]

@type_check_only
class _Tokenizer(Protocol):
    """Anything with rouge-score's tokenizer interface."""

    def tokenize(self, text: str, /) -> Sequence[str]: ...

class RougeScorer(scoring.BaseScorer):
    """Calculate rouge scores between two blobs of text (rouge-score's ``RougeScorer``).

    Sample usage:
        ```python
        scorer = RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        scores = scorer.score("The quick brown fox jumps over the lazy dog",
                              "The quick brown dog jumps on the log.")
        ```
    """

    rouge_types: Sequence[str]
    """The rouge types to calculate, read at every call (as in rouge-score)."""

    def __init__(
        self,
        rouge_types: Sequence[str],
        use_stemmer: bool = False,
        split_summaries: bool = False,
        tokenizer: _Tokenizer | None = None,
    ) -> None:
        """Initializes a new RougeScorer.

        Valid rouge types that can be computed are:

        - ``rouge1`` .. ``rouge9``: n-gram based scoring.
        - ``rougeL``: longest common subsequence based scoring.
        - ``rougeLsum``: summary-level LCS over sentences (lines, or nltk
          sentences with ``split_summaries``).

        Types are checked when scoring, as in rouge-score.

        Args:
            rouge_types: A list of rouge types to calculate.
            use_stemmer: Bool indicating whether Porter stemmer should be used to
                strip word suffixes to improve matching. This arg is used in the
                DefaultTokenizer, but other tokenizers might or might not choose to
                use this.
            split_summaries: whether to add newlines between sentences for rougeLsum
                (sentences are split with ``nltk.sent_tokenize``).
            tokenizer: Tokenizer object which has a tokenize() method. A falsy
                value means the default tokenizer.
        """

    def score_multi(self, targets: Iterable[str], prediction: str) -> dict[str, scoring.Score]:
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

    def score(self, target: str, prediction: str) -> dict[str, scoring.Score]:
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

    def score_batch(
        self,
        targets: Sequence[str] | ArrowStrings,
        predictions: Sequence[str] | ArrowStrings,
        *,
        threads: int | None = None,
    ) -> list[dict[str, scoring.Score]]:
        """Score many pairs at once, on pyroid's thread pool (not part of rouge-score).

        Element ``i`` equals ``self.score(targets[i], predictions[i])``. With the
        default tokenizer and a list/tuple of rouge type strs, every pair is
        tokenized and scored in Rust in parallel (nltk still splits sentences
        in Python with ``split_summaries=True``). Otherwise (a custom tokenizer,
        list items that are not ``str``) the pairs are scored one by one with
        :meth:`score`.

        Args:
            targets: The targets: a list or tuple of str, or an Arrow string
                array/column (``string``, ``large_string`` or ``string_view``;
                pyarrow, polars, ``pyroid.Column``, ...).
            predictions: The predictions, as for ``targets``, of the same length.
            threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at
                most ``n`` tasks at once; ``1`` runs on the calling thread.

        Returns:
            One ``dict[str, Score]`` per pair, in order.

        Raises:
            ValueError: ``targets`` and ``predictions`` differ in length, an
                Arrow value is null (ROUGE needs a str per pair: fill nulls
                first), or a rouge type is invalid (raised before any pair is
                scored, even for an empty batch).
            pyroid.InputError: ``targets`` or ``predictions`` is not a list,
                tuple or Arrow string array (a single str is rejected), or
                ``threads`` is not an int.
            UnicodeEncodeError: A str holds a lone surrogate.
            UnicodeDecodeError: An Arrow value is not valid UTF-8.
        """

    _tokenizer: Any
    _split_summaries: Any
