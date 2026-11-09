# pyroid

Text cleaning for LLM datasets, written in Rust. pyroid is a drop-in replacement for ftfy,
fastText inference, datatrove's quality filters and dedup, Trafilatura, NLTK's tokenizers, jieba,
datasketch, rouge-score, sacrebleu and json_repair, with the same API and the same output. It runs
on CPython 3.11 to 3.15, including free-threaded 3.14t and 3.15t.

!!! warning "Pre-release"
    pyroid 1.0 is not on PyPI yet. Until it is, `pip install pyroid` gives the unrelated 0.7.x
    toolkit ([Migrating from 0.7](migrating-from-0.7.md)).

## Try it without changing your code

Clean a Parquet or JSONL corpus with datatrove's FineWeb recipe (ftfy, language ID, the four
quality filters, MinHash dedup). The output is the same as datatrove's pipeline:

```console
$ pip install "pyroid[cli]"
$ pyroid clean in.parquet out.parquet --recipe fineweb
```

Or make code you don't own (a tokenizer, a language detector, a pipeline) use pyroid wherever it
imports ftfy or fastText:

```python
import pyroid

pyroid.patch("ftfy", "fasttext")  # from here on, `import ftfy` gets pyroid.ftfy

import ftfy

ftfy.fix_text("naÃ¯ve rÃ©sumÃ©")  # 'naïve résumé', computed by pyroid
```

See [the command line](guides/cli.md) and [`pyroid.patch()`](guides/patch.md).

## The problem

Building a pretraining or fine-tuning corpus means running every document through the same few
steps: fix broken Unicode, identify the language, drop low-quality pages, remove near-duplicates.
The standard tools for those steps are written in pure Python (ftfy, datatrove's filters and
MinHash) or ship as C++ extensions that lag behind new Pythons (fastText).

Profiling [datatrove](https://github.com/huggingface/datatrove)'s FineWeb chain on one core shows
where the time goes. The four quality filters take about 79% of the CPU and MinHash about 18%,
while ftfy is 2.3% and language ID 1%. These shares come from the first baseline run, in a
Docker VM ([method and caveats](benchmarks.md#where-the-cpu-goes)). The filters and MinHash both
tokenize every document with spaCy and then do their own work in Python. pyroid moves both to
Rust.

From raw WARC files there is one more step in front, extracting the text with Trafilatura, and it
costs more than all the others together: on 8 Common Crawl WARC files, stock datatrove spent 5,543
of its 6,759 CPU seconds in it (AWS EC2 c7i.4xlarge, 8 workers). pyroid ports that step too
([`pyroid.trafilatura`](guides/trafilatura.md)).

## 30-second example

```python
import pyroid.ftfy as ftfy  # was: import ftfy
import pyroid.fasttext as fasttext  # was: import fasttext
import pyroid.filters as filters  # datatrove's Gopher / C4 / FineWeb filters

docs = ["naÃ¯ve rÃ©sumÃ©", "Some other web page ..."]

# ["naïve résumé", ...]: the same as [ftfy.fix_text(d) for d in docs], on all cores
docs = ftfy.fix_text_batch(docs)

# The same .bin / .ftz files. A batch runs without the GIL: [["__label__fr"], ["__label__en"]]
model = fasttext.load_model("lid.176.bin")
labels, probs = model.predict(docs, k=1)

# datatrove's keep/drop decision per document: [False, False] (under 50 words)
keep = filters.gopher_quality(docs)
```

Batch functions take a `str`, a `list` or `tuple` of `str`, or an Arrow string column (pyarrow,
polars, pandas, Hugging Face `datasets`). Text and masks come back in the same kind. See
[Inputs and outputs](concepts/inputs.md).

## Install

```console
$ pip install pyroid               # every module but fasttext and datasketch; no dependencies
$ pip install "pyroid[fasttext]"   # adds numpy, which pyroid.fasttext returns arrays in
$ pip install "pyroid[cli]"        # the pyroid command line (pyarrow)
```

`[datatrove]` and `[hf]` add the datatrove steps and the Hugging Face helpers. There are prebuilt
wheels for Linux (glibc and musl), macOS and Windows, on x86-64 and ARM64. See
[Install](install.md).

## What's in it

| Module | Replaces | Identical output? |
|---|---|---|
| [`pyroid.ftfy`](guides/ftfy.md) | [ftfy](https://github.com/rspeer/python-ftfy) 6.3.1 | byte-identical ([parity](compatibility.md#ftfy)) |
| [`pyroid.fasttext`](guides/fasttext.md) | [fastText](https://github.com/facebookresearch/fastText) inference (`fasttext`, `fasttext-predict`, ...) | bit-identical labels and probabilities, per architecture ([parity](compatibility.md#fasttext)) |
| [`pyroid.filters`](guides/filters.md) | [datatrove](https://github.com/huggingface/datatrove) 0.10.1's Gopher, C4 and FineWeb filters, PII formatter, C4 bad words, symbol lines and URL filter | same decisions, reasons and text ([parity](compatibility.md#filters)) |
| [`pyroid.dedup`](guides/dedup.md) | datatrove 0.10.1's MinHash and sentence dedup | same signatures, removals and files ([parity](compatibility.md#dedup)) |
| [`pyroid.trafilatura`](guides/trafilatura.md) | [Trafilatura](https://github.com/adbar/trafilatura)'s text extraction, 2.3.1 or 1.11.0 (the release datatrove 0.10.1 installs), with lxml 6.1.3's HTML parser and jusText 3.0.2 | the same text, byte for byte ([parity](compatibility.md#trafilatura)) |
| [`pyroid.translit`](guides/translit.md) | `anyascii`; a permissive alternative to GPL `Unidecode` | identical to `anyascii`; not Unidecode's output |
| [`pyroid.metrics`](guides/metrics.md) | [rouge-score](https://github.com/google-research/google-research/tree/master/rouge) 0.1.2 and [sacrebleu](https://github.com/mjpost/sacrebleu) 2.6.0 (BLEU, chrF, chrF++) | identical scores and signatures, no tolerance ([parity](compatibility.md#metrics)) |
| [`pyroid.json_repair`](guides/json_repair.md) | [json_repair](https://github.com/mangiucugna/json_repair) 0.64.5 | same output, in linear time ([parity](compatibility.md#json_repair)) |
| [`pyroid.datasketch`](guides/datasketch.md) | [datasketch](https://github.com/ekzhu/datasketch) 2.0.0's MinHash, LeanMinHash, MinHashLSH, LSH Forest and LSH Ensemble | same hash values, bytes and query results ([parity](compatibility.md#datasketch)) |
| [`pyroid.jieba`](guides/jieba.md) | [jieba](https://github.com/fxsjy/jieba) 0.42.1 | same words, offsets and frequencies ([parity](compatibility.md#jieba)) |
| [`pyroid.nltk`](guides/nltk.md) | [NLTK](https://www.nltk.org) 3.10.4's `sent_tokenize` (Punkt), `word_tokenize`, Treebank and regexp tokenizers | same sentences, tokens and offsets ([parity](compatibility.md#nltk)) |

The filters and dedup follow datatrove beyond English too. 4,013 of its 4,343 language codes are
tokenized natively (Japanese with an installed SudachiDict dictionary), and the rest through
datatrove's own tokenizers ([Languages](guides/languages.md)).

| Integration | What it does |
|---|---|
| [`pyroid clean`](guides/cli.md) | datatrove's FineWeb pipeline as one command over Parquet or JSONL, with datatrove's output document for document |
| [Large datasets](guides/large-datasets.md) | the same commands over thousands of files: worker processes, `--shard I/N` across machines, `s3://`, `gs://` and `hf://` inputs, resumable runs, and MinHash dedup across shards in datatrove's file format |
| [`pyroid.patch()`](guides/patch.md) | makes `import ftfy`, `fasttext`, `anyascii`, `datasketch`, `trafilatura` and `nltk` give pyroid's functions, in code you don't own |
| [datatrove steps](guides/datatrove.md) | `pyroid.integrations.datatrove`: datatrove's pipeline steps with the same names and signatures, `Trafilatura` included, as a one-import change |
| [Hugging Face `datasets`](guides/huggingface.md) | `pyroid.integrations.hf`: batched callables for `Dataset.map` and `Dataset.filter`, and the FineWeb recipe |

## Speed

From Common Crawl WARC files to deduplicated documents, datatrove's FineWeb pipeline ran 8.4×
faster with pyroid's steps in datatrove's executor: 104.0 s against 870.8 s with 8 workers, and
775.9 s against 6,975.8 s with one (9.0×). That is 8 WARC files of CC-MAIN-2026-39 (171,595 HTML
records) on an AWS EC2 c7i.4xlarge (Intel Xeon Platinum 8488C), pinned to 8 CPUs. The kept
documents are the same once the pages stock datatrove drops at its 1 s extraction timeout are
left out too ([From WARC files](benchmarks.md#from-warc-files-datatroves-fineweb-pipeline)).

On 100,000 FineWeb documents, whose text is already extracted, pyroid's FineWeb chain is about
27× faster than datatrove on 8 processes (10.2 s against 272.1 s) and about 56× faster
single-threaded (43.4 s against 2,444.4 s), with identical output, on the same instance type. On
the 10,000-document dev slice the same ratios are 18× and 38×. On a c8g.4xlarge (AWS Graviton4),
the 100k run gives 29× and 50×.

Per stage, on the 10k dev slice and the c7i, pyroid on one thread against the incumbent in one
process:

| Module | Incumbent | Speedup |
|---|---|---:|
| `pyroid.ftfy` | ftfy 6.3.1 | 108× |
| `pyroid.fasttext` | fasttext-predict 0.9.2.4 | 1.6× (`.bin`), 3.6× (`.ftz`) |
| `pyroid.filters` | datatrove 0.10.1's Gopher, C4 and FineWeb filters | 35–69× |
| `pyroid.dedup` | datatrove 0.10.1's MinHash signatures | 34× |
| `pyroid.translit` | anyascii 0.3.3 | 53× |

The other modules have only been measured in a Docker/OrbStack VM on Apple Silicon, so their
numbers are indicative. On one thread, `pyroid.trafilatura.extract` is 7.2× faster than
Trafilatura 2.3.1 and 8.6× faster than 1.11.0, ROUGE 59–65× faster than rouge-score, corpus BLEU
11× and chrF 19× faster than sacrebleu, and `pyroid.json_repair` about 30× faster than json_repair
on broken JSON. `pyroid.nltk` splits sentences 60× and words 31× faster than NLTK, and
`pyroid.jieba` segments a batch 24× faster than jieba (58× on 8 threads).

The [Benchmarks](benchmarks.md) page has every number, how it was measured and what it doesn't
show.

## Correctness

Each module is tested differentially against the library it replaces. The inputs are pinned
FineWeb and FineWeb-2 documents, pinned Common Crawl pages, pinned evaluation data, the
incumbent's own test corpus, and Hypothesis fuzzing that includes lone surrogates, mojibake and
every option. The counts are on the [Compatibility](compatibility.md) page, and every known
difference is listed in each guide.

All suites ran on Linux aarch64 in a Docker VM. The ftfy, fastText, filters, dedup, metrics and
json_repair suites also ran natively on x86_64 and aarch64 on AWS EC2, and the Trafilatura pages
on x86_64, with 0 differences ([Platforms](compatibility.md#platforms)). macOS and Windows are
checked in CI before release.

pyroid never silently computes something different. Where a module meets an option or input it
does not port, it hands the call to the real library and says so with a
[`FallbackWarning`](concepts/fallbacks.md), or raises an error. Its results also don't depend on
the number of threads or workers or on the machine's load, where stock datatrove's Trafilatura
step drops whichever pages take more than 1 s.

## Licence

pyroid's own code is MIT. The parts ported from other projects keep their licences, so the
package's licence expression is `MIT AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause AND ISC AND
PSF-2.0 AND Unicode-3.0`:

- Apache-2.0 for the ftfy, datatrove, rouge-score, NLTK (Porter stemmer and tokenizers),
  sacrebleu, Trafilatura and SudachiPy ports;
- MIT for the spaCy, indic-nlp-library, urduhack, json_repair, datasketch, jieba (and its
  dictionary) and libxml2 (HTML parser) ports;
- BSD-2-Clause for the jusText port and its stoplists;
- BSD-3-Clause for numpy's random generator, scipy's QUADPACK integration, tldextract and lxml;
- PSF-2.0 for CPython's `ipaddress`;
- ISC for anyascii's data;
- Unicode-3.0 for Unicode Character Database data.

The [NOTICE](https://github.com/ao/pyroid/blob/main/NOTICE) file lists the ported files, and the
licence texts are in [LICENSES/](https://github.com/ao/pyroid/tree/main/LICENSES).
