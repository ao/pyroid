# pyroid

**Rust-fast text cleaning for LLM datasets.** Drop-in replacements for ftfy, fastText inference and
datatrove's quality filters: the same API, the same output, on any CPython from 3.11 to 3.15,
including free-threaded 3.14t and 3.15t.

!!! warning "Pre-release"
    pyroid 1.0 is not on PyPI yet; `pip install pyroid` still gives the unrelated 0.7.x toolkit
    (see [Migrating from 0.7](migrating-from-0.7.md)). Every number on this site is a preliminary
    measurement in a Docker VM, labelled as such; a bare-metal benchmark run is pending.

## The problem

Building a pretraining or fine-tuning corpus means running every document through the same few
steps: fix broken Unicode, identify the language, drop low-quality pages, remove near-duplicates.
The standard tools for those steps are written in pure Python (ftfy, datatrove's filters and
MinHash) or ship as C++ extensions that lag behind new Pythons (fastText).

Profiling [datatrove](https://github.com/huggingface/datatrove)'s FineWeb chain on one core shows
where the time goes: the four quality filters take about **79%** of the CPU and MinHash about
**18%**, while ftfy is 2.3% and language ID 1%
([methodology and caveats](benchmarks.md#where-the-cpu-goes)). Most of that time is spaCy
tokenization and Python string handling, which is the part pyroid moves to Rust.

## 30-second example

```python
import pyroid.ftfy as ftfy  # was: import ftfy
import pyroid.fasttext as fasttext  # was: import fasttext
import pyroid.filters as filters  # datatrove's Gopher / C4 / FineWeb filters

docs = ["âœ” No problems", "Some other web page ..."]

# ["✔ No problems", ...]: the same as [ftfy.fix_text(d) for d in docs], on all cores
docs = ftfy.fix_text_batch(docs)

# The same .bin / .ftz files. A batch runs without the GIL: [["__label__en"], ["__label__en"]]
model = fasttext.load_model("lid.176.bin")
labels, probs = model.predict(docs, k=1)

# datatrove's keep/drop decision per document: [False, False] (under 50 words)
keep = filters.gopher_quality(docs)
```

Every function takes a `str`, a `list` or `tuple` of `str`, or an Arrow string column (pyarrow,
polars, pandas, Hugging Face `datasets`), and returns the same kind. See
[Inputs and outputs](concepts/inputs.md).

## Install

```console
$ pip install pyroid               # ftfy, filters, dedup, translit; no runtime dependencies
$ pip install "pyroid[fasttext]"   # adds numpy, which pyroid.fasttext returns arrays in
```

Prebuilt wheels for Linux (glibc and musl), macOS and Windows, x86-64 and ARM64. See
[Install](install.md).

## What's in it

| Module | Replaces | Status | Identical output? |
|---|---|---|---|
| [`pyroid.ftfy`](guides/ftfy.md) | [ftfy](https://github.com/rspeer/python-ftfy) 6.3.1 | ready | byte-identical ([parity](compatibility.md#ftfy)) |
| [`pyroid.fasttext`](guides/fasttext.md) | [fastText](https://github.com/facebookresearch/fastText) inference (`fasttext`, `fasttext-predict`, ...) | ready | bit-identical labels and probabilities ([parity](compatibility.md#fasttext)) |
| [`pyroid.filters`](guides/filters.md) | [datatrove](https://github.com/huggingface/datatrove) 0.10.1 Gopher, C4 and FineWeb filters | ready | same decisions, reasons and text ([parity](compatibility.md#filters)) |
| [`pyroid.dedup`](guides/dedup.md) | datatrove 0.10.1 MinHash dedup | *preview* | being finished ([parity](compatibility.md#dedup)) |
| [`pyroid.translit`](guides/translit.md) | `anyascii`; a permissive alternative to GPL `Unidecode` | *preview* | `anyascii`-identical; *not* Unidecode's output |

How fast? Per-module numbers, how they were measured and what they don't show are on the
[Benchmarks](benchmarks.md) page. In short, single-threaded in a Docker VM: `pyroid.ftfy` is
95–167× faster than ftfy on FineWeb, `pyroid.fasttext` 1.5–5.4× faster than fastText, and
`pyroid.filters` about 86× faster than datatrove for FineWeb's four filters (*indicative,
preliminary*).

## Why trust the output?

Each module is tested differentially against the library it replaces: pinned FineWeb documents,
the incumbent's own test corpus, and Hypothesis fuzzing that includes lone surrogates, mojibake
and every option. The counts are on the [Compatibility](compatibility.md) page, and every known
difference is listed in each guide.

pyroid never silently runs something else. When a call needs a feature the Rust port does not
have, it hands the call to the original library and says so with a
[`FallbackWarning`](concepts/fallbacks.md).
