# API reference

Generated from pyroid's type stubs (the `.pyi` files shipped in the wheel), which are also what
your type checker and IDE read. The integrations have no stubs; their pages come from the Python
source.

| Module | Contents |
|---|---|
| [`pyroid`](pyroid.md) | `PyroidError`, `InputError`, `FallbackWarning`, `Column`, `patch`, `unpatch` |
| [`pyroid.ftfy`](ftfy.md) | ftfy 6.3.1's API, plus `fix_text_batch` |
| [`pyroid.fasttext`](fasttext.md) | `load_model`, `Model`, `ModelError`, `tokenize` |
| [`pyroid.filters`](filters.md) | datatrove's Gopher, C4 and FineWeb filters, `apply`, PII, C4 bad words, symbol lines, `URLFilter`, `languages` |
| [`pyroid.dedup`](dedup.md) | `minhash_dedup`, `MinHashDedup`, `sentence_dedup`, `SentenceDedup`, `exact_dedup`, `ExactDedup`, `minhash_signatures`, `simplify_text`, datatrove's config classes |
| [`pyroid.translit`](translit.md) | `anyascii`, `unidecode`, `is_ascii`, `UnidecodeError` |
| [`pyroid.metrics`](metrics.md) | `rouge_scorer`, `scoring`, `tokenizers`, `tokenize` (rouge-score's modules) and `sacrebleu` |
| [`pyroid.json_repair`](json_repair.md) | `repair_json`, `loads`, `load`, `from_file`, `repair_json_batch`, `NestingError` |
| [`pyroid.integrations`](integrations.md) | the datatrove steps, the Hugging Face helpers, the FineWeb recipe, `patch` and friends |
| [`pyroid.text`](text.md) | `normalize_whitespace`, `min_chars`: small examples of pyroid's conventions |

Batch functions are listed with one signature per input kind (`str`, `list`, `tuple`, Arrow);
their parameter tables give each parameter's types across all of them. The conventions they share are in
[Inputs and outputs](../concepts/inputs.md), [Threads](../concepts/threads.md) and
[Errors](../concepts/errors.md).
