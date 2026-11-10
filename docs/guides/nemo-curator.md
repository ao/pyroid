# NeMo Curator: drop-in text stages

`pyroid.integrations.nemo_curator` has drop-in replacements for the CPU text stages of
[NeMo Curator](https://github.com/NVIDIA-NeMo/Curator) 1.4.0 (© NVIDIA, Apache-2.0): the
`ScoreFilter`, `Score`, `Filter` and `Modify` stages, the 23 heuristic document filters, the two
fastText filters, `UnicodeReformatter` and the string modifiers. Each class subclasses NeMo's
class of the same name and takes the same arguments. The output is the same: the same scores
(values and pandas dtypes), the same kept rows, the same text, and the same exception where NeMo's
code raises one. The stages work on whole batches in Rust instead of calling a Python method once
per row through `pandas.Series.apply`.

```console
$ pip install "pyroid[nemo]"     # numpy; NeMo Curator itself is installed as NeMo documents
```

NeMo Curator runs on CPython 3.11 to 3.13. The kernels are also in [`pyroid.nemo`](#pyroidnemo),
which does not need NeMo and runs on 3.14t and 3.15t.

## Three ways to switch

Change the imports:

```diff
-from nemo_curator.stages.text.filters import ScoreFilter
-from nemo_curator.stages.text.filters.heuristic import WordCountFilter, UrlsFilter
+from pyroid.integrations.nemo_curator import ScoreFilter, WordCountFilter, UrlsFilter

 pipeline.add_stage(ScoreFilter(WordCountFilter(min_words=80)))
 pipeline.add_stage(ScoreFilter(UrlsFilter()))
```

Or only the stages. pyroid's `ScoreFilter`, `Score`, `Filter` and `Modify` also batch NeMo's own
filter and modifier objects, so in a pipeline built from NeMo's YAML configs it is enough to point
the stage targets at pyroid:

```yaml
  - _target_: pyroid.integrations.nemo_curator.ScoreFilter
    filter_obj:
      _target_: nemo_curator.stages.text.filters.heuristic.string.WordCountFilter
      min_words: 50
      max_words: 100000
```

Or patch NeMo's modules before the pipeline is built, and leave the code and the YAML alone:

```python
import pyroid

pyroid.patch("nemo_curator")   # undo with pyroid.unpatch("nemo_curator")
```

The rest of the pipeline (readers, writers, the executor, stages pyroid does not replace) stays
NeMo's. Ray workers need nothing installed besides pyroid: the stage objects they receive are
pyroid's classes.

## What is covered

| NeMo Curator 1.4.0 | pyroid |
|---|---|
| `ScoreFilter`, `Score`, `Filter`, `Modify` | the same stages, one pyroid call per batch and filter; keyword `threads` (default 1, since NeMo runs one worker per CPU; `None` for pyroid's whole thread pool) |
| `NonAlphaNumericFilter`, `SymbolsToWordsFilter`, `NumbersFilter`, `UrlsFilter`, `BulletsFilter`, `WhiteSpaceFilter`, `ParenthesesFilter`, `LongWordFilter`, `WordCountFilter`, `BoilerPlateStringFilter`, `MeanWordLengthFilter`, `PunctuationFilter`, `EllipsisFilter`, `CommonEnglishWordsFilter`, `WordsWithoutAlphabetsFilter`, `PornographicUrlsFilter`, `SubstringFilter` | [`pyroid.nemo`](#pyroidnemo) |
| `RepeatedLinesFilter`, `RepeatedParagraphsFilter`, `RepeatedLinesByCharFilter`, `RepeatedParagraphsByCharFilter`, `RepeatingTopNGramsFilter`, `RepeatingDuplicateNGramsFilter` | [`pyroid.nemo`](#pyroidnemo) |
| `FastTextLangId`, `FastTextQualityFilter` | the model is loaded by [`pyroid.fasttext`](fasttext.md) and scored in one batched prediction; `fasttext` (which needs a C++ compiler to install) is not needed |
| `UnicodeReformatter` | [`pyroid.ftfy`](ftfy.md) with NeMo's defaults (ftfy is not needed) |
| `BoilerPlateStringModifier`, `MarkdownRemover`, `NewlineNormalizer`, `QuotationRemover`, `UrlRemover`, `LineRemover`, `Slicer`, `FastTextLabelModifier` | [`pyroid.nemo`](#pyroidnemo) |

This is every CPU stage of NeMo's default text configs (`heuristic_filter_english_pipeline.yaml`,
`heuristic_filter_non_english_pipeline.yaml`, `fasttext_filter_pipeline.yaml`) and of its
modifier benchmark. Not covered: the code filters (they need sentencepiece, BeautifulSoup and
comment_parser), the model-based classifiers and token counters, and deduplication: NeMo
Curator 1.4.0's exact and fuzzy deduplication run on GPUs only (cuDF), and it has no CPU PII
stage.

The filters and modifiers of `pyroid.integrations.nemo_curator` also speed up NeMo's own stages:
their `score_document` and `modify_document` run pyroid's kernel for one document.

## What is the same, and what differs

- Scores: the same floats bit for bit, and the same pandas dtypes `Series.apply` would give for
  the batch's text column (`double[pyarrow]` for NeMo's Parquet reader, `float64` for an object
  column).
- `RepeatingTopNGramsFilter`: when several n-grams are the most frequent, NeMo takes the first in
  the order of a Python `set`, which depends on the process's string hash seed. NeMo's own
  scores for such documents (about half of FineWeb's for `n=2`) change from run to run unless
  `PYTHONHASHSEED` is fixed, and so can the kept rows. pyroid makes the same choice NeMo makes in
  the same process, so with a fixed seed both give the same output.
- `FastTextQualityFilter` draws from numpy's global random state, as NeMo does (one draw per
  document, in row order).
- Errors: NeMo's filters raise on some documents (`ZeroDivisionError` for an empty or blank one
  in most ratio filters, `ValueError` from `LongWordFilter`). The stage then runs NeMo's code on
  that batch, which raises the same exception.
- `FastTextLangId.keep_document` reads its score strings with `ast.literal_eval` where NeMo uses
  `eval`: the same values for every string `score_document` makes, and no code is run from a
  score column.
- NeMo's own code runs, with a one-time `pyroid.FallbackWarning`, for: a custom `url_regex`;
  Japanese word splitting (`lang="ja"`, MeCab); filters given precomputed `sentences`,
  `paragraphs` or `ngrams`; modifiers with several input fields. Subclasses of NeMo's or pyroid's
  filters run their own `score_document`.

## Parity

Against NeMo Curator 1.4.0 on FineWeb's dev slice and pyroid's adversarial set (13,232
documents, `parity/`, suite `nemo`): 578 checks and 3.7 million comparisons, with 0 differences:
every filter score and exception, every modifier output, the fastText scores, the stages'
DataFrames, and Hypothesis-generated text. NeMo's own pipelines, run with stock stages and with
pyroid's (YAML targets rewritten, or `pyroid.patch`), give byte-identical output. Linux aarch64,
CPython 3.12; the pinned references also pass on 3.14t and 3.15t, where NeMo does not install.
The counts are on the [Compatibility](../compatibility.md#nemo) page.

## `pyroid.nemo`

The kernels, without NeMo: one function per filter, returning what `score_document` returns, and
one per string modifier, returning what `modify_document` returns. They take a str, a list or
tuple of str, or an Arrow string array, like the rest of pyroid ([Inputs](../concepts/inputs.md)).

```python
import pyroid.nemo as nemo

nemo.word_count(["one two", "three"])               # array([2, 1])
nemo.repeating_duplicate_ngrams(docs, 5)            # NeMo's score per document
nemo.remove_markdown("**bold** [link](https://x)")  # 'bold https://x'
```

NeMo's keep rule for each score is in the function's docstring (for example, `WordCountFilter`
keeps a document when `min_words <= score <= max_words`). One difference from NeMo's code: a str
with a lone surrogate raises `UnicodeEncodeError`, as everywhere in pyroid
([Errors](../concepts/errors.md)); the classes of `pyroid.integrations.nemo_curator` run NeMo's
code for such a document instead (Arrow string columns, which NeMo's Parquet reader gives, cannot
hold one).

## Speed

Measured in Docker (OrbStack VM on a Mac, aarch64, 12 vCPUs), CPython 3.12, NeMo Curator 1.4.0,
pyroid release build; indicative, not a dedicated machine. Every timed output was first checked
equal to NeMo's. Numbers and scripts: `bench/results/orbstack-aarch64-12cpu/2026-11-10-nemo/`,
`bench/nemo_bench.py`, `bench/nemo_pipeline_bench.py`.

Per stage, one CPU, FineWeb's dev slice (10,000 documents, batches of 1,000), NeMo's stage with
NeMo's filter against pyroid's stage with the same NeMo filter (`threads=1`):

| stage | NeMo | pyroid | |
|---|---:|---:|---:|
| the 28 filter stages of `heuristic_filter_english_pipeline.yaml`, in a row | 31.4 s | 1.89 s | 16.7x |
| `RepeatingTopNGramsFilter`, n = 2, 3, 4 | 3.1 to 3.8 s | 0.22 to 0.31 s | 12 to 14x |
| `RepeatingDuplicateNGramsFilter`, n = 5 to 10 | 1.5 to 1.9 s | 0.12 s | 12 to 16x |
| the character-ratio filters (`NonAlphaNumeric`, `WhiteSpace`, `WordsWithoutAlphabets`, ...) | 0.2 to 2.3 s | 9 to 52 ms | 19 to 126x |
| `UnicodeReformatter` (NeMo's defaults) | 5.0 s | 44 ms | 114x |
| the string modifiers of NeMo's modifier benchmark | 0.04 to 1.2 s | 7 to 149 ms | 3 to 48x |
| `FastTextLangId` (lid.176.bin) | 2.06 s | 1.46 s | 1.4x |

fastText's own inference takes most of `FastTextLangId`'s time either way.

The whole YAML pipeline in NeMo's executor (`XennaExecutor`, batch mode, a Ray cluster of 1 or 8
CPUs), 100,000 FineWeb documents in 64 Parquet files, 95,774 kept by both:

| CPUs | NeMo, wall | pyroid, wall | NeMo, stage time | pyroid, stage time |
|---:|---:|---:|---:|---:|
| 1 | 377 s | 88 s | 321 s | 24 s |
| 8 | 139 s | 80 s | 537 s | 55 s |

Stage time is the sum of NeMo's per-stage `process_time` over all workers. With pyroid most of the
wall time is the executor's own: starting and stopping workers for each of the 28 stages took 65
to 75 s here, whatever the stages do, so the end-to-end gain grows with the amount of data per run.
