# Derived from rouge-score 0.1.2 (Apache-2.0), © The rouge_score Authors; modified.
"""rouge-score's scoring classes (``rouge_score.scoring``), unchanged.

``Score`` and ``AggregateScore`` are namedtuples with rouge-score's fields and
reprs; ``BootstrapAggregator`` resamples with numpy's global random state
exactly as rouge-score does, so the same ``np.random.seed`` gives the same
intervals. numpy is imported when ``aggregate`` is first called (it is not a
pyroid dependency).
"""

from __future__ import annotations

import abc
import collections
from typing import Any

from pyroid import _versions

__all__ = ["AggregateScore", "BaseScorer", "BootstrapAggregator", "Score", "fmeasure"]

_versions.check(__name__, "rouge_score", "rouge-score==0.1.2")


# collections.namedtuple bases, as in rouge-score (instances keep a __dict__).
class Score(collections.namedtuple("Score", ["precision", "recall", "fmeasure"])):  # noqa: PYI024
    """Tuple containing precision, recall, and f-measure values."""


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


class AggregateScore(collections.namedtuple("AggregateScore", ["low", "mid", "high"])):  # noqa: PYI024
    """Tuple containing confidence intervals for scores."""


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError:
        raise ImportError(
            "pyroid.metrics.scoring.BootstrapAggregator.aggregate needs numpy, which is not "
            "a pyroid dependency: pip install numpy"
        ) from None
    return np


class BootstrapAggregator:
    """Aggregates scores to provide confidence intervals.

    Sample usage:
      scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
      aggregator = Aggregator()
      aggregator.add_scores(scorer.score("one two three", "one two"))
      aggregator.add_scores(scorer.score("one two five six", "seven eight"))
      result = aggregator.aggregate()
      print result
      {'rougeL': AggregateScore(
           low=Score(precision=0.0, recall=0.0, fmeasure=0.0),
           mid=Score(precision=0.5, recall=0.33, fmeasure=0.40),
           high=Score(precision=1.0, recall=0.66, fmeasure=0.80)),
       'rouge1': AggregateScore(
           low=Score(precision=0.0, recall=0.0, fmeasure=0.0),
           mid=Score(precision=0.5, recall=0.33, fmeasure=0.40),
           high=Score(precision=1.0, recall=0.66, fmeasure=0.80))}
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
        if confidence_interval < 0 or confidence_interval > 1:
            raise ValueError("confidence_interval must be in range [0, 1]")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        self._n_samples = n_samples
        self._confidence_interval = confidence_interval
        self._scores: collections.defaultdict[Any, list[Any]] = collections.defaultdict(list)

    def add_scores(self, scores: Any) -> None:
        """Adds a sample for future aggregation.

        Args:
          scores: Dict mapping score_type strings to a namedtuple object/class
            representing a score.
        """
        for score_type, score in scores.items():
            self._scores[score_type].append(score)

    def aggregate(self) -> dict[Any, AggregateScore]:
        """Aggregates scores previously added using add_scores.

        Returns:
          A dict mapping score_type to AggregateScore objects.
        """
        np = _numpy()
        result = {}
        for score_type, scores in self._scores.items():
            # Stack scores into a 2-d matrix of (sample, measure).
            score_matrix = np.vstack(tuple(scores))
            # Percentiles are returned as (interval, measure).
            percentiles = self._bootstrap_resample(score_matrix)
            # Extract the three intervals (low, mid, high).
            intervals = tuple(scores[0].__class__(*percentiles[j, :]) for j in range(3))
            result[score_type] = AggregateScore(
                low=intervals[0], mid=intervals[1], high=intervals[2]
            )
        return result

    def _bootstrap_resample(self, matrix: Any) -> Any:
        """Performs bootstrap resampling on a matrix of scores.

        Args:
          matrix: A 2-d matrix of (sample, measure).

        Returns:
          A 2-d matrix of (bounds, measure). There are three bounds: low (row 0),
          mid (row 1) and high (row 2). Mid is always the mean, while low and high
          bounds are specified by self._confidence_interval (which defaults to 0.95
          meaning it will return the 2.5th and 97.5th percentiles for a 95%
          confidence interval on the mean).
        """
        np = _numpy()
        # Matrix of (bootstrap sample, measure).
        sample_mean = np.zeros((self._n_samples, matrix.shape[1]))
        for i in range(self._n_samples):
            sample_idx = np.random.choice(np.arange(matrix.shape[0]), size=matrix.shape[0])
            sample = matrix[sample_idx, :]
            sample_mean[i, :] = np.mean(sample, axis=0)

        # Take percentiles on the estimate of the mean using bootstrap samples.
        # Final result is a (bounds, measure) matrix.
        percentile_delta = (1 - self._confidence_interval) / 2
        q = 100 * np.array([percentile_delta, 0.5, 1 - percentile_delta])
        return np.percentile(sample_mean, q, axis=0)


def fmeasure(precision: float, recall: float) -> float:
    """Computes f-measure given precision and recall values."""
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0.0
