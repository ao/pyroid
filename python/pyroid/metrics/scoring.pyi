# Derived from rouge-score 0.1.2 (Apache-2.0), © The rouge_score Authors; modified.
# Type stubs for pyroid.metrics.scoring. The module docstring is kept here on
# purpose: the API docs are built from this stub.
"""rouge-score's scoring classes (``rouge_score.scoring``), as a drop-in.

``from pyroid.metrics import scoring`` replaces ``from rouge_score import
scoring``. The classes are ports of rouge-score 0.1.2's, line for line:

- :class:`Score` and :class:`AggregateScore` are namedtuples with the same
  fields and reprs (``Score(precision=1.0, recall=0.5, fmeasure=0.666...)``);
  they compare equal to rouge-score's.
- :class:`BootstrapAggregator` resamples with numpy's *global* random state,
  exactly as rouge-score does: after the same ``np.random.seed(...)`` it
  returns identical intervals.

Differences from rouge-score
----------------------------
- numpy is imported when :meth:`BootstrapAggregator.aggregate` is first called
  (it is not a pyroid dependency). Without it, ``aggregate`` raises
  ``ImportError`` with an install hint. rouge-score imports numpy at import.
- The classes are pyroid's own: ``isinstance(score, rouge_score.scoring.Score)``
  is false for a pyroid ``Score`` (and vice versa), though they compare equal.
"""

import abc
from collections.abc import Mapping
from typing import Any, NamedTuple

from typing_extensions import disjoint_base

__all__ = ["AggregateScore", "BaseScorer", "BootstrapAggregator", "Score", "fmeasure"]

@disjoint_base
class Score(NamedTuple):
    """Tuple containing precision, recall, and f-measure values.

    rougeL and rougeLsum return ``Score(precision=0, recall=0, fmeasure=0)``
    with ``int`` zeros when the target or prediction has no tokens, as
    rouge-score does; every other value is a ``float``.
    """

    precision: float
    recall: float
    fmeasure: float

class BaseScorer(metaclass=abc.ABCMeta):
    """Base class for Scorer objects."""

    @abc.abstractmethod
    def score(self, target: str, prediction: str) -> dict[str, Score]:
        """Calculates score between the target and prediction.

        Args:
            target: Text containing the target (ground truth) text.
            prediction: Text containing the predicted text.

        Returns:
            A dict mapping each score_type (string) to Score object.
        """

@disjoint_base
class AggregateScore(NamedTuple):
    """Tuple containing confidence intervals for scores."""

    low: Score
    mid: Score
    high: Score

class BootstrapAggregator:
    """Aggregates scores to provide confidence intervals (bootstrap resampling).

    Example:
        ```python
        from pyroid.metrics import rouge_scorer, scoring

        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"])
        aggregator = scoring.BootstrapAggregator()
        aggregator.add_scores(scorer.score("one two three", "one two"))
        aggregator.add_scores(scorer.score("one two five six", "seven eight"))
        result = aggregator.aggregate()
        # {'rouge1': AggregateScore(low=..., mid=..., high=...), 'rougeL': ...}
        ```
    """

    def __init__(self, confidence_interval: float = 0.95, n_samples: int = 1000) -> None:
        """Initializes a BootstrapAggregator object.

        Args:
            confidence_interval: Confidence interval to compute on the mean as a
                decimal.
            n_samples: Number of samples to use for bootstrap resampling.

        Raises:
            ValueError: If invalid argument is given.
        """

    def add_scores(self, scores: Mapping[str, Any]) -> None:
        """Adds a sample for future aggregation.

        Args:
            scores: Dict mapping score_type strings to a namedtuple object/class
                representing a score.
        """

    def aggregate(self) -> dict[str, AggregateScore]:
        """Aggregates scores previously added using add_scores.

        The low and high bounds are the percentiles of ``n_samples`` bootstrap
        means given by ``confidence_interval``; mid is the median of those means.
        Resampling uses ``numpy.random`` (the global state).

        Returns:
            A dict mapping score_type to AggregateScore objects.

        Raises:
            ImportError: numpy is not installed.
        """

    def _bootstrap_resample(self, matrix: Any) -> Any: ...

def fmeasure(precision: float, recall: float) -> float:
    """Computes f-measure given precision and recall values.

    ``2 * precision * recall / (precision + recall)``, or ``0.0`` when both are 0.
    """
