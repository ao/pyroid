"""Rust-backed, drop-in evaluation metrics: ROUGE (rouge-score) and BLEU/chrF (sacrebleu).

Scores are exactly the incumbents' (same floats, same result types):

- ``pyroid.metrics.rouge_scorer``, ``scoring``, ``tokenizers`` and ``tokenize``
  replace the rouge-score 0.1.2 modules of the same names:
  ``from pyroid.metrics import rouge_scorer, scoring``. ``RougeScorer.score_batch``
  scores many pairs at once on pyroid's thread pool.
- ``pyroid.metrics.sacrebleu`` replaces ``sacrebleu`` 2.6.0's BLEU and chrF API
  (``corpus_bleu``, ``sentence_bleu``, ``corpus_chrf``, ``BLEU``, ``CHRF``, ...):
  ``import pyroid.metrics.sacrebleu as sacrebleu``. ``threads=`` and
  ``sentence_bleu_batch``/``sentence_chrf_batch`` are pyroid extensions.

What the Rust ports do not cover (MeCab/SentencePiece tokenizers, bootstrap
resampling, TER, ``split_summaries=True`` sentence splitting) is handed to the
incumbent with a one-time :class:`pyroid.FallbackWarning`.

Submodules are imported on demand (``import pyroid.metrics.sacrebleu``).
"""
