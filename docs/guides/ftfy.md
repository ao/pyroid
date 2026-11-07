# `pyroid.ftfy`: drop-in ftfy

`pyroid.ftfy` is a Rust port of [ftfy](https://github.com/rspeer/python-ftfy) 6.3.1 by Robyn
Speer. It has ftfy's functions with ftfy's signatures. On the same CPython version its results
are byte-identical to ftfy 6.3.1, including the explanations and ftfy's own known failures
([parity](../compatibility.md#ftfy)).

## Change one import

```diff
-import ftfy
+import pyroid.ftfy as ftfy

 ftfy.fix_text("naÃ¯ve rÃ©sumÃ©")                  # 'naïve résumé'
 ftfy.fix_text("&lt;3", unescape_html=True)        # '<3'
 ftfy.fix_and_explain("schÃ¶n")                    # ExplainedText(text='schön', explanation=[...])
 ftfy.fix_encoding("rÃ©sumÃ©")                     # 'résumé'
```

`fix_text`, `fix_text_segment`, `fix_encoding`, `fix_and_explain`, `fix_encoding_and_explain`,
the `ftfy` alias and `TextFixerConfig` with all 16 of its fields run in Rust. ftfy's own
`TextFixerConfig` objects are accepted too, so code that builds configs with ftfy keeps working:

```python
import ftfy
import pyroid.ftfy

config = ftfy.TextFixerConfig(uncurl_quotes=False, normalization="NFKC")
pyroid.ftfy.fix_text(text, config)  # same result as ftfy.fix_text(text, config)
```

## Many documents: `fix_text_batch`

`fix_text_batch` is pyroid's addition. It runs `fix_text` over a whole batch on pyroid's thread
pool, without the GIL, and each result equals `ftfy.fix_text(doc, config, **kwargs)`.

```python
import pyroid.ftfy as ftfy

fixed = ftfy.fix_text_batch(docs)  # list in, list out
fixed = ftfy.fix_text_batch(docs, uncurl_quotes=False)
fixed = ftfy.fix_text_batch(docs, threads=4)  # at most 4 tasks at once
```

Documents that need no fixing come back as the original objects, so already-clean text costs no
copies. Lone surrogates in a `str` are fixed exactly as ftfy fixes them.

### Arrow columns

Pass a pyarrow, polars or `datasets` string column and get a
[`pyroid.Column`](../reference/pyroid.md#pyroid.Column) back, with nulls kept:

```python
import pyarrow as pa
import pyarrow.parquet as pq
import pyroid.ftfy as ftfy

table = pq.read_table("fineweb.parquet", columns=["text"])
fixed = ftfy.fix_text_batch(table["text"])  # pyroid.Column of strings
table = table.set_column(0, "text", pa.chunked_array(fixed))
```

With Hugging Face `datasets`, map over batches so pyroid sees many documents at once:

```python
ds = ds.map(lambda b: {"text": ftfy.fix_text_batch(b["text"])}, batched=True, batch_size=10_000)
```

## Delegated features

These are not ported. They are handed to the real ftfy, which must then be installed, and the
first call emits a [`pyroid.FallbackWarning`](../concepts/fallbacks.md):

- `apply_plan`, `guess_bytes`, `fix_file` and `explain_unicode` (also listed in
  `pyroid.ftfy.DELEGATED`);
- option values the Rust port does not handle: a `normalization` form other than NFC, NFD, NFKC,
  NFKD or `None`, and a non-integer `max_decode_length`.

For Arrow input, `fix_text_batch` raises `pyroid.PyroidError` for those option values instead.
The submodules `ftfy.fixes`, `ftfy.badness`, `ftfy.chardata`, `ftfy.bad_codecs` and ftfy's
command-line tool are not provided.

## Unicode versions

ftfy's output depends on the Unicode version of the running Python, through `\w`, `\d` and the
normalization of newly assigned characters. pyroid ships tables for Unicode 14.0 to 17.0 and uses
the one matching `unicodedata.unidata_version`, so it matches ftfy on every CPython from 3.11 to
3.15. On a newer Python it warns once (`RuntimeWarning`) and uses the 17.0 tables.

## Differences from ftfy

From the module's documentation (`pyroid/ftfy.pyi`):

- `max_decode_length` below 1 raises `ValueError`. ftfy loops forever on such a value.
- `fix_text` on a list or tuple raises `TypeError` (ftfy raises `AttributeError`). For other
  non-`str` inputs both raise `TypeError`, with different messages.
- `TextFixerConfig`, `ExplanationStep` and `ExplainedText` are pyroid's own classes. They have the
  same fields and defaults and compare equal to ftfy's (they are tuples), but are not instances of
  ftfy's classes. ftfy's `TextFixerConfig` objects are accepted wherever a config is.

And by design:

- An exact `str` that needs no fixing comes back as the same object (ftfy returns an equal copy
  for multi-line text).
- Delegated functions and unsupported option values need ftfy installed.
- A Python whose `unicodedata` is newer than 17.0 gets a warning and the 17.0 tables.

## Speed

On an AWS c7i.4xlarge (Intel Xeon Platinum 8488C), `fix_text_batch` took 6.70 µs per document on
one thread over the first 10,000 FineWeb documents. ftfy 6.3.1 took 724 µs, so pyroid was 108×
faster, and with 8 threads it took 2.06 µs. pyroid ran on free-threaded CPython 3.15t and ftfy on
CPython 3.13. On a c8g.4xlarge (AWS Graviton4) the speedup was 148×. See
[Benchmarks](../benchmarks.md#per-stage).

On FineWeb-2 languages, the module's earlier, preliminary run measured 14–95× single-threaded,
with Hindi the slowest (Docker/OrbStack VM on Apple Silicon, indicative;
[details](../benchmarks.md#pyroidftfy)).

In a whole datatrove pipeline ftfy is only about 2% of the CPU
([Where the CPU goes](../benchmarks.md#where-the-cpu-goes)), so a faster `fix_text` on its own
barely changes pipeline throughput. It matters where `fix_text` is the hot path: tokenizers that
call it per string (CLIP-style text encoders), web-scraping cleanup, or ftfy as a standalone pass
over a corpus.

API: [`pyroid.ftfy` reference](../reference/ftfy.md).
