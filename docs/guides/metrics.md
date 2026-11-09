# `pyroid.metrics`: rouge-score and sacrebleu

`pyroid.metrics` ports two evaluation libraries to Rust:
[rouge-score](https://pypi.org/project/rouge-score/) 0.1.2
(ROUGE-N, ROUGE-L and ROUGE-Lsum, with NLTK 3.10.4's Porter stemmer; © The rouge_score Authors,
Apache-2.0) and [sacrebleu](https://github.com/mjpost/sacrebleu) 2.6.0 (BLEU, chrF and chrF++;
© Amazon.com, Inc. or its affiliates, Apache-2.0). Scores are identical to rouge-score 0.1.2 and
sacrebleu 2.6.0: the same floats bit for bit, the same result types, `str()`, `repr()`,
`format()` output, signature strings and exceptions ([parity](../compatibility.md#metrics)).

## Change one import

For ROUGE, the four rouge-score modules (`rouge_scorer`, `scoring`, `tokenizers`, `tokenize`)
have pyroid counterparts of the same names.

```diff
-from rouge_score import rouge_scorer, scoring
+from pyroid.metrics import rouge_scorer, scoring

 scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
 target = "The quick brown fox jumps over the lazy dog"
 scorer.score(target, "The quick brown dog jumps on the log.")
 # {'rouge1': Score(precision=0.75, recall=0.6666666666666666, fmeasure=0.7058823529411765),
 #  'rougeL': Score(precision=0.625, recall=0.5555555555555556, fmeasure=0.5882352941176471)}
```

BLEU and chrF are in one module, `pyroid.metrics.sacrebleu`, with sacrebleu's top-level API.

```diff
-import sacrebleu
+import pyroid.metrics.sacrebleu as sacrebleu

 hyps = ["The cat sat on the mat.", "It rained all day in Paris."]
 refs = [["The cat sat on a mat.", "It rained in Paris all day."]]
 print(sacrebleu.corpus_bleu(hyps, refs))
 # BLEU = 34.11 92.9/58.3/20.0/12.5 (BP = 1.000 ratio = 1.000 hyp_len = 14 ref_len = 14)
 print(sacrebleu.corpus_chrf(hyps, refs, word_order=2))  # chrF2++ = 71.01
```

The metric classes live in the same module, so `from sacrebleu.metrics import BLEU, CHRF` becomes
`from pyroid.metrics.sacrebleu import BLEU, CHRF`.

## ROUGE

`RougeScorer(rouge_types, use_stemmer=False, split_summaries=False, tokenizer=None)` takes
rouge-score's arguments. The types are `rouge1` to `rouge9` (n-grams), `rougeL` (longest common
subsequence) and `rougeLsum` (summary-level LCS over sentences, one per line). They are checked
when scoring, with rouge-score's `ValueError` messages.

- `score(target, prediction)` returns a `dict[str, Score]` in `rouge_types` order. `Score` is a
  namedtuple `(precision, recall, fmeasure)`, equal to rouge-score's; rougeL and rougeLsum give
  `int` zeros when a side has no tokens, as rouge-score does.
- `score_multi(targets, prediction)` scores against several targets and keeps, per type, the
  target with the highest f-measure (the first one on ties).

```python
from pyroid.metrics import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rougeL", "rougeLsum"])
target = "The cat sat on the mat.\nIt was happy."
prediction = "It was happy.\nThe cat sat on a mat."
scores = scorer.score(target, prediction)
scores["rougeL"].fmeasure  # 0.5555555555555556
scores["rougeLsum"].fmeasure  # 0.8888888888888888: sentence order no longer matters

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"])
best = scorer.score_multi(
    ["The cat sat on the mat.", "A cat was sitting on the mat."], "The cat is on the mat."
)
best["rouge2"]  # Score(precision=0.6, recall=0.6, fmeasure=0.6)
```

Tokenizing, stemming and scoring run in Rust. The default tokenizer lowercases and keeps runs of
`a-z0-9`; with `use_stemmer=True` it stems tokens longer than 3 characters with NLTK 3.10.4's
Porter stemmer, ported to Rust, so NLTK is not needed:

```python
from pyroid.metrics import tokenizers

tokenizers.DefaultTokenizer(use_stemmer=True).tokenize("The friends had a meeting")
# ['the', 'friend', 'had', 'a', 'meet']
```

A custom `tokenizer=` (any object with a `tokenize` method) is called from Python where
rouge-score calls it, and its tokens are scored in Rust.

### Confidence intervals: `BootstrapAggregator`

`scoring.BootstrapAggregator` is rouge-score's code, and it needs numpy, which pyroid does not
depend on. numpy is imported on the first `aggregate()` call, and without it that call raises
`ImportError` with an install hint. It resamples with numpy's global random state, so after the
same `np.random.seed(...)` it returns the same intervals as rouge-score.

```python
import numpy as np
from pyroid.metrics import rouge_scorer, scoring

pairs = [
    ("The cat sat on the mat.", "The cat is on the mat."),
    ("A dog barked at the mailman.", "The dog barked loudly."),
    ("It rained all day in Paris.", "It rained in Paris."),
]
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"])
np.random.seed(0)
aggregator = scoring.BootstrapAggregator()
for target, prediction in pairs:
    aggregator.add_scores(scorer.score(target, prediction))
result = aggregator.aggregate()  # {'rouge1': AggregateScore(low=..., mid=..., high=...), ...}
print(result["rouge1"].mid.fmeasure)  # 0.7444444444444445
```

## BLEU and chrF

`pyroid.metrics.sacrebleu` has `corpus_bleu`, `sentence_bleu`, `raw_corpus_bleu`, `corpus_chrf`,
`sentence_chrf`, the `BLEU` and `CHRF` classes (`corpus_score`, `sentence_score`,
`get_signature`, `references=`), the score and signature classes, `extract_word_ngrams`,
`extract_char_ngrams` and `__version__` (`"2.6.0"`). References are sacrebleu's streams:
`refs[k][i]` is the k-th reference of hypothesis `i`, and `None` marks a missing one.

```python
import pyroid.metrics.sacrebleu as sacrebleu

hyps = ["The cat sat on the mat.", "It rained all day in Paris."]
refs = [
    ["The cat sat on a mat.", "It rained in Paris all day."],
    ["A cat sat on the mat.", None],  # a second reference, missing for the second hypothesis
]
bleu = sacrebleu.BLEU()
print(bleu.corpus_score(hyps, refs))
# BLEU = 65.80 100.0/75.0/50.0/50.0 (BP = 1.000 ratio = 1.000 hyp_len = 14 ref_len = 14)
print(bleu.get_signature())  # nrefs:var|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0

print(sacrebleu.sentence_bleu("The cat sat on the mat.", ["The cat sat on a mat."]))
# BLEU = 48.89 85.7/66.7/40.0/25.0 (BP = 1.000 ratio = 1.000 hyp_len = 7 ref_len = 7)
```

These run natively:

- the tokenizers `13a` (the default), `intl`, `zh` (also `trg_lang="zh"`), `char` and `none`;
- every smoothing method;
- `lowercase`, `force`, `effective_order` and `max_ngram_order`;
- chrF and chrF++ with any `char_order`, `word_order`, `beta`, `whitespace` and `eps_smoothing`.

Rust computes the integer n-gram statistics. The final formulas (`BLEU.compute_bleu`,
`CHRF._compute_f_score`) are sacrebleu's Python, so the floats are sacrebleu's on every CPython.
sacrebleu's log warnings go to the same `sacrebleu` logger.

## Many pairs: batches and threads

These are pyroid's additions (keyword-only `threads=`, see [Threads](../concepts/threads.md)):

| Call | What it does |
|---|---|
| `RougeScorer.score_batch(targets, predictions, *, threads=None)` | element `i` equals `score(targets[i], predictions[i])`; takes lists, tuples or Arrow string columns |
| `threads=` on `corpus_bleu`, `raw_corpus_bleu`, `corpus_chrf` and `corpus_score` | computes the corpus statistics on pyroid's thread pool |
| `sentence_bleu_batch(hyps, refs, ..., *, threads=None)` | element `i` equals `sentence_bleu(hyps[i], [r[i] for r in refs], ...)` |
| `sentence_chrf_batch(hyps, refs, ..., *, threads=None)` | the same for `sentence_chrf` |

```python
import pyarrow as pa
import pyroid.metrics.sacrebleu as sacrebleu
from pyroid.metrics import rouge_scorer

targets = ["The cat sat on the mat.", "It rained all day in Paris."]
predictions = ["The cat is on the mat.", "It rained in Paris."]
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"])
scorer.score_batch(targets, predictions)  # [{'rouge1': Score(...), 'rouge2': Score(...)}, ...]
scorer.score_batch(pa.array(targets), pa.array(predictions), threads=8)  # the same list

refs = [["The cat sat on a mat.", "It rained in Paris all day."]]
scores = sacrebleu.sentence_bleu_batch(predictions, refs)  # one BLEUScore per hypothesis
sacrebleu.corpus_chrf(predictions, refs, threads=8)  # threads only changes the speed
```

ROUGE needs a `str` per pair, so a null in an Arrow column raises `ValueError`; fill nulls first.
Pass the references of the `*_batch` functions as streams (`[refs]`). A flat list of `str` raises
[`pyroid.InputError`](../reference/pyroid.md#pyroid.InputError).

## Delegated features

These are not ported. They are handed to the real library, with a one-time
[`pyroid.FallbackWarning`](../concepts/fallbacks.md) per function and reason. If the library is
not installed, the call raises [`pyroid.PyroidError`](../reference/pyroid.md#pyroid.PyroidError)
with an install hint instead.

- ROUGE: a custom tokenizer whose tokens are not lists of `str` (scored by rouge-score).
- sacrebleu: the `ja-mecab`, `ko-mecab`, `spm`, `flores101`, `flores200` and `spBLEU-1K`
  tokenizers (and `trg_lang` `"ja"` or `"ko"`), `corpus_score(..., n_bootstrap > 1)`, `TER`,
  `corpus_ter`, `sentence_ter`, `DATASETS` and the test-set helpers (listed in
  `pyroid.metrics.sacrebleu.DELEGATED`).

```python
bleu = sacrebleu.BLEU()
bleu.corpus_score(hyps, refs, n_bootstrap=1000)  # sacrebleu's result, and:
# FallbackWarning: pyroid.metrics.sacrebleu.BLEU.corpus_score: n_bootstrap > 1 is not supported
# by the Rust port; delegating to sacrebleu (slow path). This warning is shown once.
```

Importing `pyroid.metrics.sacrebleu` never imports sacrebleu; only a delegated call does. This
matters on free-threaded CPython, where `import sacrebleu` loads lxml, which turns the GIL back
on.
Likewise, `pyroid.metrics.rouge_scorer` does not need NLTK. With `split_summaries=True`,
rougeLsum's sentences are split by [`pyroid.nltk`](nltk.md)'s Punkt port, which reads NLTK's
`punkt_tab` model and imports NLTK, if installed, only to find it (rouge-score always imports
nltk, absl, numpy and six).

## Lone surrogates

A lone surrogate (U+D800 to U+DFFF) in a target, prediction, hypothesis or reference raises
`UnicodeEncodeError`, with a note naming the item, as everywhere in pyroid
([Lone surrogates](../concepts/inputs.md#lone-surrogates)). rouge-score treats them as
separators and sacrebleu as ordinary code points. Replace them first with
`s.encode("utf-8", "replace").decode("utf-8")`.

## Differences from rouge-score

From the modules' documentation. `pyroid/metrics/rouge_scorer.pyi`:

- Lone surrogates (U+D800..U+DFFF) in a `str` target or prediction raise `UnicodeEncodeError`
  (with a note) when the default tokenizer is used. rouge-score treats them as separators.
  Remove them first, e.g. `s.encode("utf-8", "replace").decode("utf-8")`. A custom tokenizer's
  tokens may hold them (those are scored by rouge-score).
- `RougeScorer` is pyroid's class, a `pyroid.metrics.scoring.BaseScorer`; its scores are
  `pyroid.metrics.scoring.Score` namedtuples, equal to rouge-score's but not instances of
  `rouge_score.scoring.Score`.
- rouge-score's private helpers (`_score_lcs`, `_lcs_table`, `_summary_level_lcs`, ...) and
  `lcs_ind` are not provided.
- `score_multi` picks the first maximal f-measure as `np.argmax` does, without numpy (its error
  for no targets is numpy's `ValueError("attempt to get argmax of an empty sequence")`).
- "Using default tokenizer." is logged at INFO level with `logging` (logger
  `pyroid.metrics.rouge_scorer`), not with absl.
- nltk is not needed. With `split_summaries=True` it is imported, if installed, to find the
  `punkt_tab` model; rouge-score always imports nltk, absl, numpy and six.

`pyroid/metrics/scoring.pyi`:

- numpy is imported when `BootstrapAggregator.aggregate` is first called (it is not a pyroid
  dependency). Without it, `aggregate` raises `ImportError` with an install hint. rouge-score
  imports numpy at import.
- The classes are pyroid's own: `isinstance(score, rouge_score.scoring.Score)` is false for a
  pyroid `Score` (and vice versa), though they compare equal.

`pyroid/metrics/tokenizers.pyi`:

- `DefaultTokenizer` has no `_stemmer` attribute (rouge-score keeps an nltk `PorterStemmer`
  there). Stemming runs in Rust.
- A `str` with lone surrogates (U+D800..U+DFFF) raises `UnicodeEncodeError` (rouge-score drops
  them like any other non-alphanumeric character).

`pyroid/metrics/tokenize.pyi`:

- A `str` with lone surrogates (U+D800..U+DFFF) raises `UnicodeEncodeError`.
- A `stemmer` passed here is called from Python for every token longer than 3, as in
  rouge-score. `pyroid.metrics.tokenizers.DefaultTokenizer` with `use_stemmer=True` stems in Rust
  instead.

## Differences from sacrebleu

From the module's documentation (`pyroid/metrics/sacrebleu.pyi`):

- Lone surrogates (`"\ud800"`) in a hypothesis or reference raise `UnicodeEncodeError` (with a
  note naming the item), as everywhere in pyroid. sacrebleu treats them as code points. Remove
  them first, e.g. `s.encode("utf-8", "replace").decode("utf-8")`.
- Wrong item types that sacrebleu's own argument checks let through (a non-str hypothesis after
  the first, a non-str, non-`None` reference) raise `pyroid.InputError` (a `TypeError`) naming
  the item. sacrebleu raises an `AttributeError` from deep inside. sacrebleu's checks themselves
  (`_check_*_args`) raise the same `TypeError`s with the same messages.
- `BLEU.tokenizer` is a pyroid tokenizer object for the native tokenizers (`signature()` and
  `__call__(line)`, the same output as sacrebleu's), and sacrebleu's own tokenizer object for the
  delegated ones.
- Pre-computed references are held in a native object. Changing `max_ngram_order`, `char_order`
  or `word_order` of a metric after constructing it with `references=` raises `ValueError`
  (sacrebleu would mix the old reference n-grams with new hypothesis ones).
- `sentence_bleu_batch` logs sacrebleu's `effective_order` recommendation once per call, not
  once per sentence.

## Parity

About 1.39 million comparisons per interpreter, of exact `repr`, `str`, `format()`, JSON and
signature strings with no tolerance, found 0 differences on CPython 3.12, 3.13, 3.14t and 3.15t.
The data is real evaluation data, pinned by sha256:

- CNN/DailyMail 3.0.0 test: 11,490 lead-3 summaries against the highlights, and 2,000 whole
  articles;
- SummEval: 1,600 summaries from 16 systems against 11 references each, through `score_multi`;
- WMT21 newstest2021: 8 language pairs, 141 systems, 163,000 segments, up to 3 references.

The Porter stemmer also matches NLTK on all 512,136 distinct tokens of a 100k-document FineWeb
sample.

sacrebleu's own BLEU scores depend on the C library's `exp` and `log`, so the stored references
are pinned per platform. On an AWS c7i.4xlarge, x86_64 gave the same results as aarch64. Under
Rosetta 2 emulation, 165 of 694,757 BLEU results differed by 1–2 ulps. pyroid matches sacrebleu
in each case. Details: [Compatibility](../compatibility.md#metrics).

## Speed

Wall time for 100,000 pairs, on a Docker/OrbStack VM on Apple Silicon, indicative. CPython 3.13,
release wheel. ROUGE runs on CNN/DailyMail lead-3 summaries against the highlights, with rouge1,
rouge2, rougeL and rougeLsum; the 100k pairs are the 11,490 pairs repeated. BLEU and chrF run on
WMT21 newstest2021 segments with the first reference and default settings. "8 processes" runs the
reference library in 8 forked processes, against pyroid at `threads=8` in one process:

| Workload, 100k pairs (ms) | Reference, 1 process | pyroid, `threads=1` | Speedup | Reference, 8 processes | pyroid, `threads=8` | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| ROUGE (`score_batch`) | 178,492 | 3,039 | 59× | 34,741 | 1,179 | 29× |
| ROUGE with the stemmer | 306,034 | 4,715 | 65× | 47,777 | 1,359 | 35× |
| corpus BLEU | 7,827 | 681 | 11× | 1,716 | 276 | 6× |
| corpus chrF | 27,476 | 1,421 | 19× | 5,213 | 288 | 18× |
| `sentence_bleu_batch` | 9,741 | 1,862 | 5× | 2,434 | 1,327 | 2× |
| `sentence_chrf_batch` | 22,661 | 1,987 | 11× | 4,361 | 915 | 5× |

One call at a time, `RougeScorer.score` took 21 µs per pair against rouge-score's 852 µs (2.3 ms
with the stemmer). `sentence_bleu` took 20 µs against 127 µs, and `sentence_chrf` 28 µs against
281 µs. Per-sentence BLEU and chrF are dominated by building sacrebleu's Python score objects,
which pyroid keeps for exactness. Other work was loading the VM, so the 8-thread figures
understate the scaling. Every timed result was first checked identical to the reference's.
Results: `bench/results/orbstack-aarch64-12cpu/2026-11-06-metrics/`; see also
[Benchmarks](../benchmarks.md#other-modules-languages-and-integrations).

API: [`pyroid.metrics` reference](../reference/metrics.md).
