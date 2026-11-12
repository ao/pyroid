# Hugging Face `datasets`

`pyroid.integrations.hf` has functions to pass to `Dataset.map` and `Dataset.filter` with
`batched=True`: ftfy, fastText language ID, the Gopher, C4 and FineWeb quality filters, and the
whole FineWeb recipe with MinHash dedup. On Arrow-format datasets pyroid reads the text column in
place, without building a Python `str` per document. `hf.fineweb(ds)` keeps the same documents,
with the same text, as datatrove 0.10.1's FineWeb pipeline
([parity](../compatibility.md#integrations)). Tested with datasets 3.6, 4.8 and 5.1.

```console
$ pip install "pyroid[hf]"     # datasets, pyarrow, numpy
```

## Change one line

```diff
-ds = ds.map(lambda doc: {"text": ftfy.fix_text(doc["text"])})
+ds = ds.with_format("arrow").map(hf.fix_text(), batched=True)
```

The pieces of the FineWeb recipe, one call each:

```python
import datasets

from pyroid.integrations import hf

ds = datasets.Dataset.from_parquet("sample.parquet").with_format("arrow")
ds = ds.map(hf.fix_text(), batched=True)  # ftfy on the text column
ds = ds.map(hf.langid("lid.176.bin"), batched=True)  # + language, language_score
ds = ds.filter(hf.quality_mask(), batched=True)  # Gopher, C4 and FineWeb filters
ds.num_rows  # 170 of the 200 documents
```

## The functions

Each returns a callable for `batched=True`:

| Function | For | What the callable does |
|---|---|---|
| `fix_text(column="text", config=None, *, threads=None, **kwargs)` | `map` | replaces `column` with `ftfy.fix_text` of it ([`pyroid.ftfy`](ftfy.md)); `config` and `**kwargs` as for `fix_text_batch` |
| `langid(lid_model=None, column="text", *, k=1, threads=None, language_column="language", score_column="language_score")` | `map` | adds fastText's top label (without `__label__`) and its probability; with `k` other than 1 (`-1`: all labels), list columns, best first |
| `quality_mask(column="text", filters=None, *, threads=None)` | `filter` | keeps what passes the quality filters, run in order as datatrove does (`pyroid.filters.apply`); `filters=None` is FineWeb's four |
| `annotate(column="text", *, lid_model=None, threads=None)` | `map` | the FineWeb chain but the dedup: sets the cleaned text and the columns `keep`, `reason`, `language`, `language_score` |
| `keep_mask(column="keep")` | `filter` | keeps the rows whose boolean column is true (e.g. `annotate`'s `keep`) |

Each document goes to fastText as one line (newlines become spaces), as in datatrove's
`LanguageFilter`. `quality_mask` uses only the filters' decision, so C4's line removal is not
applied to the text. `annotate` and `fineweb` give the text as it leaves the chain. `threads` is
pyroid's `threads=` ([Threads](../concepts/threads.md)), and `None` uses the whole thread pool for
each batch.

`annotate`'s `reason` is `None` for a kept document, else `"lid"`,
`"<stage>:<datatrove reason>"` or `"null"`, as in
[`pyroid clean`](cli.md#dropped-rows-and-filter_reason):

```python
ds = datasets.Dataset.from_parquet("sample.parquet")
ann = ds.with_format("arrow").map(hf.annotate(lid_model="lid.176.bin"), batched=True)
ann.with_format(None).select_columns(["keep", "reason"])[:2]
# {'keep': [False, True], 'reason': ['gopher_quality:gopher_below_alpha_threshold', None]}
kept = ann.filter(hf.keep_mask(), batched=True)
kept.num_rows  # 170
```

## Arrow batches and Python batches

The callables take both kinds of batch that `datasets` passes:

- With Arrow batches (`ds.with_format("arrow")`, the zero-copy path), the callable gets a
  `pyarrow.Table` and pyroid reads the text column in place. Measured on a 3 MB batch, `fix_text`
  peaked at 932 bytes of Python heap, with no growth of pyarrow's memory pool. A map callable
  returns a `pyarrow.Table` with every column of the batch, which `datasets` uses in place of the
  batch. So `remove_columns=` has no effect; call `.remove_columns()` on the dataset instead. A
  filter callable returns a boolean `pyarrow.ChunkedArray`.
- With Python batches (the default format), the callable gets the dict of lists. It returns a
  dict of the new or changed columns, which `datasets` merges into the batch, or a `list[bool]`
  for `filter`.

`None` texts stay `None`, and filters drop them (an Arrow null in a mask drops the row too). Don't
pass `input_columns=`: the callables read their column by name.

```python
ds = datasets.Dataset.from_dict({"text": ["Ã©tÃ© 2024", None, "plain"]})
ds = ds.map(hf.fix_text(), batched=True)  # a Python-format dataset
ds.to_dict()  # {'text': ['été 2024', None, 'plain']}
```

The callables also work in a streaming `IterableDataset`'s `map` and `filter`. An Arrow-format
`IterableDataset.filter` needs datasets 5; older versions reject it, whatever the function.
`fineweb` needs a `Dataset`, because MinHash reads the survivors back.

## The whole recipe: `fineweb(ds)`

`hf.fineweb` runs FineWeb's recipe on a `Dataset`, MinHash included:

1. `ds.map(annotate(...), batched=True)` on Arrow batches: ftfy, language ID (English,
   probability above 0.65) and the four quality filters;
2. MinHash with datatrove's FineWeb settings over the survivors, read back in dataset order. Each
   cluster of near-duplicates keeps one document, as in datatrove, and the others get
   `reason="minhash"`. This step runs in this process over the whole dataset, so its result does
   not depend on `batch_size` or `num_proc`.

```python
import datasets

from pyroid.integrations import hf

res = hf.fineweb(datasets.Dataset.from_parquet("sample.parquet"), lid_model="lid.176.bin")
res.stats.docs, res.stats.kept  # (200, 170)
res.stats.dropped
# {'lid': 0, 'gopher_repetition': 0, 'gopher_quality': 27, 'c4_quality': 0,
#  'fineweb_quality': 3, 'minhash': 0}
res.stats.reasons
# {'gopher_quality:gopher_below_alpha_threshold': 24, 'fineweb_quality:line_punct_ratio': 3,
#  'gopher_quality:gopher_too_many_end_ellipsis': 3}
res.dataset.to_parquet("clean.parquet")
```

It returns a `FineWebResult`:

- `dataset`: the kept documents, in order, with the cleaned text, the input's other columns and
  `language` / `language_score`;
- `annotated`: every document, with the text as it left the chain and the columns `keep` (the
  final decision) and `reason` (`"minhash"` for a near-duplicate);
- `stats`: documents in and kept, the drops per stage (in chain order) and per reason, and wall
  times (`annotate`, `dedup`, `total`); `print(res.stats)` gives a summary.

Both datasets get the input's format back. `**map_kwargs` go to step 1's `Dataset.map`
(`keep_in_memory`, `load_from_cache_file`, `cache_file_name`, `desc`, ...). An empty dataset
raises `ValueError`. `annotate` and `fineweb` run the recipe object that `pyroid clean` uses,
[`pyroid.integrations.recipes.FineWeb`](cli.md#from-python-pyroidintegrationsrecipes), which you
can also call on Arrow batches yourself.

## Caching and `num_proc`

The callables pickle by their settings. So `datasets` can fingerprint them, reusing a cached
result when you run the same map again, and ship them to `num_proc` worker processes, each of
which loads the fastText model itself. A loaded `pyroid.fasttext.Model` can't be pickled, so pass
the model's path. pyroid already runs every batch on all cores, so `num_proc` rarely helps. If you
use it, `threads` of about `os.cpu_count() // num_proc` avoids oversubscription.

`lid_model=None` finds the model as the command line does: `$PYROID_LID_MODEL`, else a cached
`lid.176.bin`, downloaded once and sha256-checked
([details](cli.md#the-language-id-model)).

## The notebook

[`examples/notebooks/clean_fineweb_in_60_seconds.ipynb`](https://github.com/ao/pyroid/blob/main/examples/notebooks/clean_fineweb_in_60_seconds.ipynb)
streams the first 10,000 documents of FineWeb's `sample-10BT`, runs `hf.fineweb` on them, and
prints the counts per stage and per reason, the time it took and a few dropped documents per
stage. Then it writes the kept documents to Parquet. It is ready for Colab.

To run it on local files without network, set `PYROID_NOTEBOOK_DATA` (a FineWeb Parquet file),
`PYROID_LID_MODEL`, and optionally `PYROID_NOTEBOOK_DOCS` and `PYROID_NOTEBOOK_OUT`. pyroid's test
suite executes it that way, headless, and checks the counts against datatrove's pipeline. See the
notebook's [README](https://github.com/ao/pyroid/blob/main/examples/notebooks/README.md).

## The Gradio app

[`examples/space/`](https://github.com/ao/pyroid/tree/main/examples/space) is a minimal Gradio
app, ready for a Hugging Face Space but not deployed. On the **Paste text** tab you see what each
stage of the chain decides for one document: whether ftfy changed it, the language and its
probability, and which filter dropped it and why. On the **Upload a file** tab you run the whole
recipe with `hf.fineweb` on a Parquet or JSONL file with a `text` column, and download the
cleaned file. Every input is capped for a public demo (characters, upload size, rows). Run
locally, it serves on <http://127.0.0.1:7860> only:

```console
$ pip install -r requirements.txt
$ python app.py
```

## Speed

The callables call pyroid's batch functions on each `datasets` batch, so they run at the
modules' speed. See [`pyroid.ftfy`](ftfy.md#speed),
[`pyroid.fasttext`](fasttext.md#speed-and-memory), [`pyroid.filters`](filters.md#speed) and
[`pyroid.dedup`](dedup.md#speed). This page has no separate timings for them.

API: [`pyroid.integrations.hf` reference](../reference/integrations.md#hf).
