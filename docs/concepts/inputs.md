# Inputs and outputs

Every batch function in pyroid follows the same rules, so once you know them for one module you
know them for all of them.

## What goes in

| You pass | Examples |
|---|---|
| one `str` | `"some text"` |
| a `list` or `tuple` of `str` | `docs`, `tuple(docs)` |
| an Arrow string column | a pyarrow `Array` or `ChunkedArray`, a polars `Series`, a pandas 3 string column, a Hugging Face `datasets` column, nanoarrow, arro3, duckdb results, or a [`pyroid.Column`](../reference/pyroid.md#pyroid.Column) |

Arrow input is anything with `__arrow_c_stream__` (preferred) or `__arrow_c_array__` (the
[Arrow PyCapsule interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html))
whose type is `string`, `large_string` or `string_view`, in one chunk or many, sliced or not.
Strings are never copied on the way in, and pyroid needs no Arrow library to read them.

`None` inside a list is an error; nulls in an Arrow column are fine (see below).

## What comes out

The result has the same kind as the input. There are three kinds of function:

| Kind | Examples | Returns |
|---|---|---|
| **Transforms** | `ftfy.fix_text_batch`, `filters.c4_lines`, `translit.anyascii` | the new text: `str` → `str`, `list` → `list`, `tuple` → `tuple`, Arrow → `pyroid.Column` of strings (`large_string` for `large_string` input, else `string`) |
| **Predicates** | `filters.gopher_quality`, `dedup.minhash_dedup`, `text.min_chars` | a mask: `bool`, `list[bool]`, `tuple[bool, ...]`, or a `pyroid.Column` of bools |
| **Other values** | `filters.gopher_quality_reasons`, `fasttext.Model.predict` | one value per document, as each function documents |

```python
import pyarrow as pa
import pyroid.text as t

t.normalize_whitespace("  a \t b ")  # 'a b'
t.normalize_whitespace(["a  b", "ok"])  # ['a b', 'ok']
t.min_chars(("abc", "x"), 2)  # (True, False)
col = t.normalize_whitespace(pa.array(["a  b", None]))  # pyroid.Column, type 'string'
col.to_pylist()  # ['a b', None]
```

A few details that matter for large corpora:

- **Unchanged documents are the original objects.** If a transform leaves a document as it was,
  you get back the very same `str` (`out[i] is docs[i]`), so nothing is copied for the common
  case of already-clean text.
- **Masks, not filtered lists.** Predicates never build new strings; use the mask to select rows
  in whatever you are using (`[d for d, k in zip(docs, keep) if k]`,
  `table.filter(pa.chunked_array(keep))`, `df.filter(pl.Series(keep))`).
- **Arrow nulls stay null.** A null document gives a null string from a transform and a null
  bool from a predicate.

## `pyroid.Column`

Arrow input gives a [`pyroid.Column`](../reference/pyroid.md#pyroid.Column): a small, read-only,
dependency-free Arrow column. It has `len()`, `.type`, `.name` (carried over from the input
field), `.null_count`, `.num_chunks` and `.to_pylist()`, and it implements the PyCapsule
interface, so any Arrow library takes it without a copy:

```python
pa.chunked_array(col)  # pyarrow
pa.array(col)  # pyarrow, one chunk (concatenates if needed)
pl.Series(col)  # polars
pyroid.text.min_chars(col, 10)  # or straight back into pyroid
```

!!! tip "polars and column names"
    polars takes the series name from the Arrow stream and ignores the name in
    `pl.Series("new", col)`. Use `pl.Series(col).alias("new")`.

Output chunks of type `string` that would pass 2 GiB are split into several chunks.
`OverflowError` is raised only for a single document over 2 GiB, or when `__arrow_c_array__`
has to concatenate past 2 GiB (use `pa.chunked_array(col)` instead).

## Lone surrogates

A Python `str` can hold lone surrogates (`"\ud800"`), which are not valid Unicode and cannot be
encoded to UTF-8. Most pyroid functions raise `UnicodeEncodeError` for them, with a note naming
the document. Where the incumbent handles them, pyroid does too: `pyroid.ftfy` fixes them exactly
as ftfy does, `pyroid.json_repair` repairs text with them as json_repair does, and
`pyroid.translit.unidecode` warns and drops them as Unidecode does. `pyroid.metrics` raises,
where rouge-score and sacrebleu accept them. Each function's reference entry says which.
