# `pyroid.filters`: datatrove's quality filters

`pyroid.filters` ports four of [datatrove](https://github.com/huggingface/datatrove) 0.10.1's
filters to Rust: **`GopherQualityFilter`, `GopherRepetitionFilter`, `C4QualityFilter` and
`FineWebQualityFilter`** (datatrove is © Hugging Face, Apache-2.0). They keep and drop **the same
documents** as datatrove, give the same `filter_reason` strings, and leave the same text after
C4's line removal ([parity](../compatibility.md#filters)).

The hard part is the words. datatrove tokenizes English with spaCy 3.8.18 (`spacy.blank("en")`
plus the sentencizer), and every word count, ratio and n-gram in these filters depends on
spaCy's tokens. pyroid includes an exact port of that tokenizer: spaCy's prefix, suffix, infix
and URL rules, its 1,347 tokenizer exceptions, the special-case matcher and the sentencizer.

## From a datatrove pipeline

datatrove filters are pipeline steps that read `Document` objects. pyroid's are functions over a
batch of texts that return masks, so they fit wherever you have the texts: a list, a Parquet
column, a `datasets` batch.

```diff
-from datatrove.pipeline.filters import GopherQualityFilter
-pipeline = [reader, GopherQualityFilter(min_doc_words=100), writer]
+import pyroid.filters as F
+keep = F.gopher_quality(texts, min_doc_words=100)   # one bool per text: datatrove's decision
```

To run several filters, chain them in one call with `apply`, which runs the filters in datatrove's pipeline order:
a document dropped by one filter is not seen by the next, C4's rewritten text is what later
filters see, and steps that see the same text tokenize it once.

```python
import pyroid.filters as F

result = F.apply(texts, [
    F.GopherRepetition(),
    F.GopherQuality(),
    F.C4Quality(filter_no_terminal_punct=False),   # FineWeb's C4 settings
    F.FineWebQuality(),
])
result.keep      # keep mask, e.g. [True, False, ...]
result.reason    # datatrove's filter_reason, None when kept, e.g. [None, 'dup_line_frac', ...]
result.text      # each document's text after the chain (C4 removes lines)
kept = [t for t, k in zip(result.text, result.keep) if k]
```

You can pass datatrove's own filter objects instead of pyroid's configs; their settings are read
from them:

```python
from datatrove.pipeline.filters import GopherQualityFilter
F.apply(texts, [GopherQualityFilter(min_doc_words=100)])
```

## One filter at a time

Each filter has three functions, all with datatrove's parameter names and defaults (as keyword
arguments, or a config object such as `F.GopherQuality(...)`):

| Function | Returns, per document |
|---|---|
| `gopher_quality(texts, **params)` | `True` if datatrove keeps it |
| `gopher_quality_reasons(texts, **params)` | datatrove's drop reason, or `None` when kept |
| `gopher_repetition`, `gopher_repetition_reasons` | the same, for `GopherRepetitionFilter` |
| `c4_quality`, `c4_quality_reasons`, `c4_lines` | the same for `C4QualityFilter`; `c4_lines` returns the text after C4's line removal |
| `fineweb_quality`, `fineweb_quality_reasons` | the same, for `FineWebQualityFilter` |

```python
F.gopher_quality(texts, min_doc_words=100)
F.gopher_repetition_reasons(texts, dup_line_frac=0.25)
F.c4_lines(texts, F.C4Quality(filter_no_terminal_punct=False))
```

A falsy threshold (`None`, `0`) disables its check, as in datatrove. Bad parameter types and
values are checked when the call starts, before any document is filtered.

## Arrow, `datasets` and threads

Predicates return a mask in the input's kind, and Arrow input gives a
[`pyroid.Column`](../reference/pyroid.md#pyroid.Column) (nulls stay null):

```python
import pyarrow as pa
import pyarrow.parquet as pq
import pyroid.filters as F

table = pq.read_table("fineweb.parquet")
keep = F.fineweb_quality(table["text"], threads=8)   # pyroid.Column of bools
table = table.filter(pa.chunked_array(keep))
```

With Hugging Face `datasets`:

```python
ds = ds.filter(lambda b: F.gopher_quality(b["text"]), batched=True, batch_size=10_000)
```

## Languages

`language` accepts every datatrove language code that maps to `SpaCyTokenizer("en")`:
`"eng"` (the default) and `"en"`, the proxies `"frr"`, `"fry"`, `"fy"`, `"sco"` and `"stq"`, and
each of them with a `_Latn` suffix (`F.english_languages()` lists them). Other languages raise `NotImplementedError`. Python string
semantics (`isalpha`, `isspace`, `splitlines`, `lower`, `re`'s `\w` and `\d`) follow the running
interpreter's Unicode version, as datatrove's do.

## Differences from datatrove

From the module's documentation (`pyroid/filters.pyi`):

- A `top_n_grams` / `dup_n_grams` size below 1 raises `ValueError` (datatrove loops forever on
  `dup_n_grams` sizes below 1).
- Values are checked when the call starts: a non-number threshold raises `TypeError` before any
  document is filtered; a failing document raises for the whole batch (the exception names the
  document).
- `exclusion_writer` is not supported (use the `*_reasons` functions).
- A document with a token that is exactly `IS_ALPHA` (e.g. in `"IS_ALPHA."`, not in
  `"xIS_ALPHA"`) is tokenized like any other text, as datatrove does the first time in a
  process. After that, spaCy fails on such documents (its string store maps `IS_ALPHA` to id 1,
  which is its hash map's deleted-key marker), and datatrove re-tokenizes
  `text.replace("IS_ALPHA", "")` instead, or raises if that still has the token. So datatrove's
  words for these documents depend on what it saw before; pyroid's do not.

Also: on a Python newer than Unicode 17.0, pyroid warns once and uses its 17.0 tables. Other
datatrove filters (`C4BadWordsFilter`, `C4ParagraphFilter`, ...) are not ported. pyroid never
delegates to datatrove, so it never emits `FallbackWarning`.

## Speed

On one core, FineWeb's four filters through `apply` measured about 86× faster than datatrove
(0.28 ms against 23.9 ms per document), and single filters 30–105×.
*Docker/OrbStack VM on Apple Silicon, indicative, preliminary*: see
[Benchmarks](../benchmarks.md#pyroidfilters).

API: [`pyroid.filters` reference](../reference/filters.md).
