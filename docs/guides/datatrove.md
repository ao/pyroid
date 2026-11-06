# datatrove: drop-in pipeline steps

`pyroid.integrations.datatrove` has drop-in replacements for 15 of
[datatrove](https://github.com/huggingface/datatrove) 0.10.1's pipeline steps (datatrove is
© Hugging Face, Apache-2.0). Each one **subclasses datatrove's own step of the same name**, takes
the same constructor arguments and gives the same documents: the same `text`, the same
`metadata`, the same keep/drop decisions and `filter_reason`s, so exclusion writers and later
steps see the same data. The work runs in Rust, in batches, without the GIL. With the steps
swapped in, datatrove's FineWeb pipeline gives the **same per-document digest** as the stock
pipeline in datatrove's own executor ([parity](../compatibility.md#integrations)).

```console
$ pip install "pyroid[datatrove]"     # datatrove, regex, tokenizers, numpy
```

## Change one import

```diff
-from datatrove.pipeline.filters import GopherQualityFilter, LanguageFilter
+from pyroid.integrations.datatrove import GopherQualityFilter, LanguageFilter

 pipeline = [
     reader,
     LanguageFilter(exclusion_writer=JsonlWriter("removed/lid")),
     GopherQualityFilter(exclusion_writer=JsonlWriter("removed/gopher")),
     writer,
 ]
```

The rest of the pipeline (readers, writers, the executor, steps pyroid does not replace) stays
datatrove's, and the steps mix freely.

## The steps

| pyroid step | datatrove's step | Runs on |
|---|---|---|
| `FTFYFormatter` | `datatrove.pipeline.formatters.FTFYFormatter` | [`pyroid.ftfy`](ftfy.md), byte-identical to ftfy 6.3.1 (ftfy isn't needed) |
| `PIIFormatter` | `datatrove.pipeline.formatters.PIIFormatter` | pyroid's email and IP matchers ([`filters.md`](filters.md)); datatrove's own replacement counters and IP validator |
| `SymbolLinesFormatter` | `datatrove.pipeline.formatters.SymbolLinesFormatter` | [`pyroid.filters`](filters.md) |
| `LanguageFilter` | `datatrove.pipeline.filters.LanguageFilter` | one batched [`pyroid.fasttext`](fasttext.md) prediction per batch; datatrove's own `filter()` code makes the decision and the metadata |
| `URLFilter` | `datatrove.pipeline.filters.URLFilter` | [`pyroid.filters.URLFilter`](filters.md), with this step's own `tldextract` suffix list and datatrove's block lists (pyahocorasick isn't needed) |
| `GopherRepetitionFilter` | `datatrove.pipeline.filters.GopherRepetitionFilter` | [`pyroid.filters`](filters.md) (spaCy isn't needed) |
| `GopherQualityFilter` | `datatrove.pipeline.filters.GopherQualityFilter` | the same |
| `C4QualityFilter` | `datatrove.pipeline.filters.C4QualityFilter` | the same, with C4's line removal |
| `FineWebQualityFilter` | `datatrove.pipeline.filters.FineWebQualityFilter` | the same |
| `C4BadWordsFilter` | `datatrove.pipeline.filters.C4BadWordsFilter` | pyroid's word search; datatrove's word lists (loaded as datatrove loads them) and its random draws for `keep_fraction` |
| `MinhashDedupSignature` | `datatrove.pipeline.dedup.MinhashDedupSignature` (MinHash stage 1) | [`pyroid.dedup`](dedup.md) signatures, written to the same files, byte for byte |
| `SentenceDedupSignature` | `datatrove.pipeline.dedup.SentenceDedupSignature` (stage 1) | [`pyroid.dedup`](dedup.md); the same `.c4_sig` files, byte for byte |
| `SentenceFindDedups` | `datatrove.pipeline.dedup.SentenceFindDedups` (stage 2) | a merge in Rust; the same `.c4_dup` files |
| `SentenceDedupFilter` | `datatrove.pipeline.dedup.SentenceDedupFilter` (stage 3) | [`pyroid.dedup`](dedup.md); the same documents and exclusion-writer documents |
| `SentenceDedupBuildIndex` | `datatrove.pipeline.dedup.sentence_dedup.SentenceDedupBuildIndex` | a merge in Rust; the same index file |

MinHash stages 2 to 4 (`MinhashDedupBuckets`, `MinhashDedupCluster`, `MinhashDedupFilter`) are
not replaced: keep datatrove's, which read pyroid's signature files unchanged. They were 2 to 3%
of the pipeline's wall time with pyroid's steps at 8 workers. To deduplicate in memory instead of
through files, use [`pyroid.dedup`](dedup.md) directly; the native APIs of every step are in
[`filters.md`](filters.md) and [`dedup.md`](dedup.md).

The drop-ins need fewer packages than datatrove's steps: no spaCy for the filters, MinHash and
sentence dedup (for the languages pyroid tokenizes, [below](#languages)), no ftfy, no
`fasttext-numpy2-wheel`, and no `fasteners` for `LanguageFilter` when `model_path` is given.
`URLFilter` needs tldextract and fasteners, as datatrove's does.

## Extra arguments

Every step takes datatrove's constructor arguments plus these keyword-only ones
(`inspect.signature` and `help()` show them):

| Argument | Default | Meaning |
|---|---|---|
| `batch_size` | `1000` | documents per pyroid call |
| `threads` | `1` | pyroid's `threads=` ([Threads](../concepts/threads.md)). datatrove already runs one task per process, so each step uses one thread by default; pass `None` for pyroid's whole thread pool when you run few tasks |
| `model_path` | `None` | `LanguageFilter` only: a local `lid.176.bin` (or a GlotLID `.bin` with `backend="glotlid"`); `None` downloads the model into datatrove's cache, as datatrove does |

`SentenceFindDedups` and `SentenceDedupBuildIndex` are one sequential merge each; they accept
`batch_size` and `threads` for uniformity.

## A small run

datatrove steps are callables over a stream of documents, so you can try them without an
executor:

```python
from datatrove.data import Document

from pyroid.integrations.datatrove import (
    FTFYFormatter,
    GopherQualityFilter,
    LanguageFilter,
    PIIFormatter,
)

story = (
    "The town council met on Tuesday evening to discuss the plans for the new library. "
    "Residents asked about the cost, the opening hours and the parking, and the mayor said "
    "that the building would open in the spring of next year if the weather allowed it. "
    "Questions can be sent to the clerk at clerk@example-town.org before the next meeting, "
    "which will be held in the school hall on the first Monday of the month."
)
docs = [
    Document(text=story.replace("town council", "townâ€™s council"), id="0"),
    Document(text="Too short to keep.", id="1"),
]
gopher = GopherQualityFilter()
steps = [FTFYFormatter(), PIIFormatter(), LanguageFilter(model_path="lid.176.bin"), gopher]
data = iter(docs)
for step in steps:
    data = step(data)  # a datatrove step is a callable over a stream of documents
kept = list(data)
[d.id for d in kept]  # ['0']
kept[0].text[:30]  # 'The town’s council met on Tues'
"email@example.com" in kept[0].text  # True: PIIFormatter's first replacement
kept[0].metadata  # {'language': 'en', 'language_score': 0.9667896628379822}
gopher.stats.to_dict()["stats"]
# {'batches': 1, 'total': 2, 'forwarded': 1, 'doc_len': {'total': 405, 'mean': 405.0},
#  'dropped': 1, 'dropped_gopher_short_doc': 1}
```

## A full pipeline

[`examples/datatrove/fineweb.py`](https://github.com/ao/pyroid/blob/main/examples/datatrove/fineweb.py)
is datatrove's FineWeb pipeline after text extraction (`examples/fineweb.py` in datatrove's
repository), on pyroid's steps: `LanguageFilter`, the Gopher repetition and quality, C4 and
FineWeb quality filters, each with an exclusion writer, then MinHash with FineWeb's settings
(`MinhashDedupSignature` from pyroid, stages 2 to 4 from datatrove), in datatrove's
`LocalPipelineExecutor`. The only difference from the stock pipeline is the import block;
`--stock` runs datatrove's own steps, for comparison.

```console
$ pip install "pyroid[datatrove]"
$ python fineweb.py data/ out/ --tasks 2 --workers 2 --lid-model lid.176.bin
```

The input is a folder of Parquet or JSONL files with a `text` column; the kept documents end up
in `out/output`, the dropped ones in `out/removed/<stage>` with their `filter_reason`.
[`examples/datatrove/verify.py`](https://github.com/ao/pyroid/blob/main/examples/datatrove/verify.py)
runs the benchmark's chain with the steps swapped in and checks its digest (it needs a pyroid
checkout and the benchmark data).

## Languages

The steps take datatrove's `language` argument, a datatrove language code such as `"fra_Latn"`.
Every step that tokenizes follows the same rule:

- `GopherRepetitionFilter`, `GopherQualityFilter`, `C4QualityFilter`, `FineWebQualityFilter`,
  `MinhashDedupSignature`, `SentenceDedupSignature` and `SentenceDedupFilter` run in Rust for
  every code pyroid tokenizes natively. `pyroid.filters.languages()` marks those codes `native`:
  3,999 of datatrove's 4,343, about 77% of FineWeb-2's bytes. They include English, the spaCy
  rule-based languages (German, French, Spanish, Russian, Arabic, ...), the IndicNLP languages
  (Hindi, Bengali, Urdu, ...) and the `WhitespaceTokenizer` ones. Each code gets the word and
  sentence tokenizer datatrove assigns it, reproduced exactly, so neither spaCy nor
  indic-nlp-library is needed.
- Any other `language` runs datatrove's own code for that step, with one
  [`pyroid.FallbackWarning`](../concepts/fallbacks.md) per step and language. That covers the codes
  whose datatrove tokenizer pyroid does not port (Chinese, Japanese, Vietnamese, Thai, the Stanza
  languages, ...), a code datatrove has no tokenizer for, and a `WordTokenizer` object. The code
  that runs then needs the language's segmenter installed, as datatrove does. The sentence-dedup
  steps warn only when their configuration needs a tokenizer. `SentenceFindDedups` and
  `SentenceDedupBuildIndex` only merge files and take no `language`.
- `C4BadWordsFilter` uses each document's `language` metadata to pick datatrove's word list, as
  datatrove does; `LanguageFilter`, `URLFilter`, `PIIFormatter`, `SymbolLinesFormatter` and
  `FTFYFormatter` do not depend on a tokenizer.

```python
from pyroid.integrations.datatrove import GopherQualityFilter, MinhashDedupSignature

GopherQualityFilter(language="fra_Latn")  # in Rust: spaCy's French rules
MinhashDedupSignature(output_folder="sigs", language="hin_Deva")  # in Rust: IndicNLP's tokenizer

step = GopherQualityFilter(language="cmn_Hani")  # datatrove's code (jieba)
# At its first batch:
# FallbackWarning: pyroid.integrations.datatrove.GopherQualityFilter: language='cmn_Hani' is not
# supported by pyroid; running datatrove's own code (slow path). This warning is shown once.
```

The [Languages](languages.md) guide lists the codes and their tokenizers, and what takes
`language=` in pyroid's own APIs.

## Differences from datatrove

From the module's documentation (`pyroid/integrations/datatrove.py`):

- Per-line statistics of `C4QualityFilter` (`line-total`, `line-filter-*`) are not counted, and
  batched steps also count `batches`.
- A language that pyroid does not tokenize natively runs datatrove's own code for that step,
  with a one-time `pyroid.FallbackWarning`. The same goes for a code datatrove has no tokenizer
  for, and for a `WordTokenizer` object (see [Languages](#languages)).

And from the steps' documentation:

- `MinhashDedupSignature`: datatrove's sort pass logs "Found existing sig files ... Skipping sig
  writing step": pyroid has just written them.
- `C4BadWordsFilter`: a batch that contains a document whose language has no list raises before
  any of the batch is yielded (datatrove yields the documents before it first).
- `URLFilter`: a document without a `url` raises datatrove's `AssertionError` when its batch is
  filtered (before any document of that batch is passed on); datatrove's per-document
  `DeprecationWarning` from tldextract's `registered_domain` (hidden by Python's default filters)
  is not emitted; `block_listed_domains` and the other list attributes hold the constructor's
  entries only (the integrated lists live in Rust); `banned_subwords_automaton` is `None`.
- `SentenceDedupBuildIndex`: like datatrove's own step, it fails (`StopIteration`) on an empty
  signature file; that case runs datatrove's code.

Also:

- Errors are raised before a batch is processed, where datatrove raises mid-stream: a bad-words
  language with no list (with `fail_on_missing_language`), an empty URL, an empty replacement
  tuple. The same goes for the errors datatrove's tokenizers raise, which pyroid reproduces: for
  example, `C4QualityFilter` on Hindi raises datatrove's `ValueError("substring not found")` for
  the documents where IndicNLP's sentence splitter rewrites the text.
- Urdu sentences (`C4QualityFilter`) follow `datatrove[multilingual]`, where indic-nlp imports
  urduhack. With plain datatrove and indic-nlp-library, datatrove's own step raises
  `ModuleNotFoundError`.
- The steps' timing statistics are per batch, not per document.
- With a datatrove other than 0.10.1 installed, importing the module gives one `RuntimeWarning`:
  the steps reproduce 0.10.1's decisions.
- datatrove's quirks are reproduced, not fixed: for instance, sentence dedup's stage 1 strips
  sentences while stage 3 doesn't, so window indices can shift, and a blank `URLFilter` list
  entry drops every URL without a registered domain.

## Parity

- **Whole pipeline:** datatrove's FineWeb chain in its own `LocalPipelineExecutor` with 8
  workers, with `FTFYFormatter`, `LanguageFilter`, the four quality filters and
  `MinhashDedupSignature` swapped for pyroid's, gives the same per-document digest (id, text,
  language, score, keep, reason, duplicate flag) as the stock pipeline: `ef314b8d…` on the
  10,000-document FineWeb dev slice and `c0c6b746…` on the 13,232-document parity corpus (dev plus
  adversarial documents).
- **Per step, against live datatrove** (67 tests on 313 documents): `FTFYFormatter` with 3
  configs at batch sizes 1, 7 and 1,000; the four filters with 8 configs at 3 batch and thread
  settings (same result, reason, rewritten text and metadata); `LanguageFilter` with 7 configs;
  `MinhashDedupSignature`'s files byte-identical, and datatrove's stages 2 to 4 on them remove the
  same documents.
- **The newer steps:** `PIIFormatter` (4 configs), `C4BadWordsFilter` (5) and
  `SymbolLinesFormatter` (3) on the 13,232 documents; `URLFilter` on the 10,000 dev URLs with 7
  configs; sentence dedup on the 13,232 documents with 7 configs and an 8,300-document corpus
  with injected duplicates with 9 configs (signature, duplicate and index files byte-identical,
  the same documents and exclusion documents). All with 0 differences, plus property-based tests
  against live datatrove.
- **Per language, in datatrove's executor**
  ([`examples/datatrove/verify_fineweb2.py`](https://github.com/ao/pyroid/blob/main/examples/datatrove/verify_fineweb2.py)):
  each FineWeb-2 slice with two near-duplicate copies (750 documents) through three filter chains
  (FineWeb-2's own per-language settings; datatrove's defaults with C4; the same without C4),
  then FineWeb-2's MinHash (xxhash, 64 bits) with the language, gives the same per-document
  digest as the stock pipeline for German, French, Spanish, Russian, Arabic and Hindi (in Rust)
  and for Chinese and Japanese (datatrove's own code). Where stock datatrove fails, pyroid fails
  the same way, in the same phase: C4 on Hindi raises `ValueError("substring not found")` in
  both. The timing runs below gave the same digests again, on 2,000 and 5,000 documents.
- **Per language, per step, against live datatrove:** the four quality filters with 5
  configurations on 6 native languages (spaCy's rules, IndicNLP, `WhitespaceTokenizer`), with no
  `FallbackWarning`; `MinhashDedupSignature`'s files byte-identical on the same languages and on
  Chinese (fallback).

See [Compatibility](../compatibility.md#integrations).

## Speed

With the steps swapped in, datatrove's FineWeb pipeline on the FineWeb dev slice (10,000
documents, datatrove's executor, 8 processes) took 21.1 s and 10.2 s in two runs, against 498.8 s
and 213.4 s for the stock pipeline run in the same minute: **21–24× less wall time** (23.6× and
20.9×; 25.7× and 22.9× less CPU time). In one process on the first 2,000 documents it was 17.6×
and 16.7× faster. *Docker/OrbStack VM on Apple Silicon (12 vCPU), indicative*: the VM was
heavily shared (load average 18 to 30), which inflates stock datatrove most, so each ratio comes
from a same-minute pair (stock, then pyroid). pyroid's times include datatrove's executor
phases, worker start-up, Parquet IO and the 125 MiB model load. On a quieter VM, the
consolidated benchmark measured 14.7× for pyroid's stages in datatrove's executor at 8
processes (2.85 s against 41.93 s; [Benchmarks](../benchmarks.md#end-to-end-the-fineweb-chain)),
which is the better guide to an unloaded machine.

Per language, FineWeb-2's own pipeline (its step-5 filters with the language's settings, then
MinHash with the language: all five executor phases, Parquet in and out) on each FineWeb-2 slice
with near-duplicate copies, in datatrove's executor, stock and with the steps swapped in.
*Docker/OrbStack VM on Apple Silicon (12 vCPU), indicative*: release build, CPython 3.13, and a
shared VM (load average 10 to 28), so each ratio comes from a back-to-back pair; at 8 processes
the second pair ran pyroid first.

| Language | 1 process, 2,000 documents: datatrove / pyroid | Speedup | 8 processes, 5,000 documents: datatrove / pyroid, two pairs | Speedup |
|---|---:|---:|---:|---:|
| German (`deu_Latn`) | 36.3 s / 2.21 s | 16× | 25.9 s / 1.97 s; 27.5 s / 2.48 s | 13×; 11× |
| French (`fra_Latn`) | 75.0 s / 2.70 s | 28× | 66.3 s / 5.71 s; 45.0 s / 2.47 s | 12×; 18× |
| Spanish (`spa_Latn`) | 50.8 s / 2.36 s | 22× | 31.5 s / 2.55 s; 17.8 s / 2.10 s | 12×; 8.5× |
| Russian (`rus_Cyrl`) | 65.2 s / 2.49 s | 26× | 31.4 s / 2.11 s; 38.3 s / 1.94 s | 15×; 20× |
| Arabic (`arb_Arab`) | 44.6 s / 2.69 s | 17× | 21.5 s / 1.67 s; 24.8 s / 4.16 s | 13×; 6.0× |
| Hindi (`hin_Deva`, IndicNLP) | 10.5 s / 1.94 s | 5.4× | 9.2 s / 3.38 s; 5.5 s / 1.36 s | 2.7×; 4.1× |

pyroid's times are mostly datatrove's executor. In one process, pyroid's filter phase took 0.7
to 1.3 s (22 to 35× faster than datatrove's; Hindi 9.3×) and its signature phase 0.6 to 0.8 s
(21 to 39×; Hindi 5.7×), and datatrove's MinHash stages 2 to 4, which both runs share, took
another 0.6 to 0.9 s. Hindi gains least because datatrove's IndicNLP tokenizer is already cheap.
Chinese and Japanese run datatrove's own code in both pipelines, at datatrove's speed.

The newer steps, on the FineWeb dev slice (*Docker/OrbStack VM on Apple Silicon (12 vCPU),
indicative*; medians of 5 runs for pyroid and 2 for datatrove, load average about 8; datatrove's
8 processes are a warmed `multiprocessing.Pool(8)`):

| Step | datatrove, 1 process | datatrove, 8 processes | pyroid, 1 thread | pyroid, 8 threads | Speedup, 1 / 8 threads (vs 8 processes) |
|---|---:|---:|---:|---:|---:|
| PIIFormatter, 10,000 documents | 2.885 s | 0.595 s | 0.032 s | 0.019 s | 89× / 31× |
| C4BadWordsFilter, 10,000 documents | 6.968 s | 1.312 s | 0.305 s | 0.053 s | 23× / 25× |
| SymbolLinesFormatter (FineWeb-2's settings), 10,000 documents | 0.145 s | 0.198 s | 0.028 s | 0.008 s | 5.2× / 25× |

These time datatrove's step alone and pyroid's native call. For the drop-in steps themselves, in
runs on the same VM: `URLFilter.run()` passed 242k documents/s against datatrove's 27k (load
average about 22), and sentence dedup's stages 1 and 3 ran at 5,664 documents/s on one thread and
10,457 on eight, against datatrove's 98 in one process (58× and 107×, on the 13,232 documents).
See
[Benchmarks](../benchmarks.md#more-datatrove-steps) and
[the whole pipeline](../benchmarks.md#datatroves-fineweb-pipeline-with-pyroids-steps-swapped-in).

API: [`pyroid.integrations.datatrove` reference](../reference/integrations.md#datatrove).
