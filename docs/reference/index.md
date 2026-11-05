# API reference

Generated from pyroid's type stubs (the `.pyi` files shipped in the wheel), which are also what
your type checker and IDE read.

| Module | Contents |
|---|---|
| [`pyroid`](pyroid.md) | `PyroidError`, `InputError`, `FallbackWarning`, `Column` |
| [`pyroid.ftfy`](ftfy.md) | ftfy 6.3.1's API, plus `fix_text_batch` |
| [`pyroid.fasttext`](fasttext.md) | `load_model`, `Model`, `ModelError`, `tokenize` |
| [`pyroid.filters`](filters.md) | datatrove's Gopher, C4 and FineWeb filters, `apply` |
| [`pyroid.text`](text.md) | `normalize_whitespace`, `min_chars`: small examples of pyroid's conventions |

`pyroid.dedup` and `pyroid.translit` are in preview; their reference pages appear here once they
are merged (until then, see their guides: [dedup](../guides/dedup.md),
[translit](../guides/translit.md)).

Batch functions are listed with one signature per input kind (`str`, `list`, `tuple`, Arrow);
their parameter tables give each parameter's types across all of them. The conventions they share are in
[Inputs and outputs](../concepts/inputs.md), [Threads](../concepts/threads.md) and
[Errors](../concepts/errors.md).
