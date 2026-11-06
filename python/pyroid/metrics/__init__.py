"""Rust-backed, drop-in evaluation metrics: ROUGE (rouge-score) and BLEU/chrF (sacrebleu).

Scores are exactly the incumbents' (same floats, same result types):

- ``pyroid.metrics.rouge_scorer`` and ``pyroid.metrics.scoring`` replace
  ``rouge_score.rouge_scorer`` and ``rouge_score.scoring`` (rouge-score 0.1.2):
  ``from pyroid.metrics import rouge_scorer, scoring``.
- ``pyroid.metrics.sacrebleu`` replaces ``sacrebleu`` 2.6.0's BLEU and chrF API:
  ``import pyroid.metrics.sacrebleu as sacrebleu``.

Submodules are imported on demand (``import pyroid.metrics.sacrebleu``).
"""
