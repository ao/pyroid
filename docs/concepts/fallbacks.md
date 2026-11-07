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
| `pyroid.metrics` | ROUGE: tokens from a custom tokenizer that are not lists of `str` (to rouge-score), and `split_summaries=True`'s sentence split (to `nltk.sent_tokenize`; scoring stays native). sacrebleu: the `ja-mecab`, `ko-mecab`, `spm`, `flores101`, `flores200` and `spBLEU-1K` tokenizers, `n_bootstrap > 1`, TER, `DATASETS` and the test-set helpers (`pyroid.metrics.sacrebleu.DELEGATED`) | — |
| `pyroid.json_repair` | `schema=` calls, non-`str` input, and text holding a character in U+10F800..U+10FFFF (the range the port uses to store lone surrogates) | `repair_json_batch` on Arrow input raises `PyroidError` for those characters instead of delegating (pass a list) |
| `pyroid.integrations.datatrove` | a `language` the step can't tokenize natively runs datatrove's own code for that step | — |

Without the library a call needs (ftfy, datatrove and the language's segmenter, rouge-score,
nltk, sacrebleu, json_repair), it raises `pyroid.PyroidError` and says what to install.

## `python_tokenizer=True`

datatrove 0.10.1 assigns every language code a word tokenizer. pyroid reproduces the rule-based
ones in Rust: spaCy's tokenizer for 57 languages, IndicNLP and the whitespace tokenizer, which
cover 3,999 of datatrove's 4,343 codes. The other 344 codes use statistical or dictionary
segmenters (jieba for Chinese, SudachiPy for Japanese, pyvi, Stanza, Thai, Khmer and others) that
pyroid does not port. For those, pass `python_tokenizer=True` to the filters or to MinHash:

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
imports ftfy, fastText or anyascii get pyroid's functions instead. It announces that with its
own `PatchWarning`, once per process for each patched name, so you can tell the two apart.
