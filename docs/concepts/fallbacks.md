# Fallbacks: `pyroid.FallbackWarning`

pyroid's modules port the incumbent library's behaviour to Rust, but not always all of it. When a
call needs something the port does not handle, pyroid does not guess and does not quietly
compute something close. It does one of two things:

1. It delegates: it calls the original library, returns that library's result, and emits
   [`pyroid.FallbackWarning`](../reference/pyroid.md#pyroid.FallbackWarning), once per process for
   each function and reason.
2. It raises `pyroid.PyroidError` when the original library is not installed, saying what was
   needed.

So the result is always the incumbent's. The warning tells you that this call took the slow path.

## Which modules delegate

| Module | Delegates | Otherwise |
|---|---|---|
| `pyroid.ftfy` | `apply_plan`, `guess_bytes`, `fix_file`, `explain_unicode` (listed in `pyroid.ftfy.DELEGATED`), and option values the port does not handle: a `normalization` form other than NFC/NFD/NFKC/NFKD/`None`, a non-integer `max_decode_length` | `fix_text_batch` on Arrow input raises `PyroidError` for those options instead of delegating |
| `pyroid.fasttext` | never | — |
| `pyroid.filters` | with `python_tokenizer=True`, the tokenization of a language pyroid does not tokenize natively (see below); the filter math stays in Rust. On that path a `C4Quality` step runs as datatrove's own `C4QualityFilter` | such a language without `python_tokenizer=True` raises `NotImplementedError`; other unsupported parameters raise (e.g. `exclusion_writer`) |
| `pyroid.dedup` | MinHash with `python_tokenizer=True`, as for the filters: datatrove's tokenizer splits the words, everything else stays in Rust | without it, such a language raises `NotImplementedError`; sentence dedup has no Python-tokenizer path |
| `pyroid.translit` | never | — |
| `pyroid.metrics` | ROUGE: tokens from a custom tokenizer that are not lists of `str` (to rouge-score). sacrebleu: the `ja-mecab`, `ko-mecab`, `spm`, `flores101`, `flores200` and `spBLEU-1K` tokenizers, `n_bootstrap > 1`, TER, `DATASETS` and the test-set helpers (`pyroid.metrics.sacrebleu.DELEGATED`) | — |
| `pyroid.json_repair` | `schema=` calls, non-`str` input, and text holding a character in U+10F800..U+10FFFF (the range the port uses to store lone surrogates) | `repair_json_batch` on Arrow input raises `PyroidError` for those characters instead of delegating (pass a list) |
| `pyroid.trafilatura` | the other output formats, `include_formatting`, `include_links`, `include_images`, metadata (`with_metadata`, `url_blacklist`, `extract_with_metadata`, ...), `target_language`, `prune_xpath`, `options=`, a trafilatura `LRUCache`, an lxml tree or HTTP response as input, bytes that are compressed or not UTF-8 ([the list](../guides/trafilatura.md#delegated-options)) | the installed trafilatura must be the mode's release (2.3.1, or 1.11.0 with `compat="1.11.0"`); without it, or with another version, these calls raise `PyroidError` |
| `pyroid.nltk` | Punkt training (`PunktTrainer`, `PunktSentenceTokenizer(train_text)`), a custom `token_cls`, a `PunktLanguageVars` subclass that changes more than `sent_end_chars`, parameters that are not sets of `str`, `dump()` and saving parameters, a Treebank tokenizer whose patterns were changed, a `RegexpTokenizer` pattern that is not one of the native ones | — |
| `pyroid.jieba` | `pyroid.jieba.posseg`, `pyroid.jieba.analyse`, paddle mode, and `get_dict_file()` for the default dictionary | — |
| `pyroid.datasketch` | the classes pyroid does not port (`bBitMinHash`, `HNSW`, `MinHashLSHBloom`, `WeightedMinHash`, `HyperLogLog`, ...), Redis and Cassandra storage, and a `hashfunc` other than `sha1_hash32`/`sha1_hash64` (called from Python for each token; the permutations stay in Rust) | — |
| `pyroid.integrations.datatrove` | a `language` the step can't tokenize natively runs datatrove's own code for that step, and `Trafilatura` with an installed trafilatura release that pyroid does not reproduce runs datatrove's own step | — |

Without the library a call needs (ftfy, datatrove and the language's segmenter, rouge-score,
nltk, sacrebleu, json_repair, trafilatura, jieba, datasketch), it raises `pyroid.PyroidError` and
says what to install.

## `python_tokenizer=True`

datatrove 0.10.1 assigns every language code a word tokenizer. pyroid reproduces the rule-based
ones in Rust: spaCy's tokenizer for 57 languages, IndicNLP and the whitespace tokenizer, plus
spaCy's Chinese tokenizer with jieba, datatrove's Japanese tokenizer with SudachiPy (which reads
an installed SudachiDict dictionary), spaCy's Vietnamese tokenizer with pyvi and datatrove's Thai
tokenizer with PyThaiNLP (which reads an installed PyThaiNLP), which together cover 4,049 of
datatrove's 4,343 codes. The other 294 codes use statistical or dictionary segmenters (Stanza,
Kiwi, Khmer and others) that pyroid does not port. For those, and for Japanese without a
dictionary or Thai without PyThaiNLP, pass
`python_tokenizer=True` to the filters or to MinHash:

- each document is tokenized by datatrove's own tokenizer, in Python, on the calling thread
  (datatrove 0.10.1 and the segmenter must be installed);
- the filter or MinHash math then runs in Rust on those words, so the decisions are datatrove's;
- the call emits one `FallbackWarning` per function and language.

That path runs at the segmenter's speed and doesn't scale with `threads=`, so with several cores
datatrove's own processes can be faster. The [Languages](../guides/languages.md) guide lists
which codes are native and how to choose.

## Handling the warning

```python
import warnings
import pyroid

# Silence the slow-path notice everywhere:
warnings.simplefilter("ignore", pyroid.FallbackWarning)

# Or make it an error in tests, to be sure every call runs in Rust:
warnings.simplefilter("error", pyroid.FallbackWarning)
```

`FallbackWarning` is a `RuntimeWarning`. pyroid always emits that exact class for delegation,
so a filter on it never hides anything else.

Other warnings are about results, not code paths, and stay plain `RuntimeWarning`s. For example,
a Python whose `unicodedata` is newer than the Unicode tables pyroid ships (17.0) gets a one-time
`RuntimeWarning`, and pyroid uses its 17.0 tables. Any other warning matches the incumbent's own
(for example `pyroid.translit.unidecode`'s lone-surrogate warning).

## `pyroid.patch()` warns too

[`pyroid.patch()`](../guides/patch.md) is not a fallback but the opposite: it makes code that
imports ftfy, fastText, anyascii, datasketch, trafilatura or NLTK get pyroid's functions instead.
It announces that with its own `PatchWarning`, once per process for each patched name, so you can
tell the two apart.
