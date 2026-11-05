# Fallbacks: `pyroid.FallbackWarning`

pyroid's modules port the incumbent library's behaviour to Rust, but not always all of it. When a
call needs something the port does not handle, pyroid does **not** guess and does not quietly
compute something close. It does one of two things:

1. **Delegate:** call the original library, return *its* result, and emit
   [`pyroid.FallbackWarning`](../reference/pyroid.md#pyroid.FallbackWarning), once per process for
   each function and reason.
2. **Raise `pyroid.PyroidError`** when the original library is not installed, saying what was
   needed.

So the result is always the incumbent's; the warning tells you that this call took the slow path.

## Which modules delegate

| Module | Delegates | Otherwise |
|---|---|---|
| `pyroid.ftfy` | `apply_plan`, `guess_bytes`, `fix_file`, `explain_unicode` (listed in `pyroid.ftfy.DELEGATED`), and option values the port does not handle: a `normalization` form other than NFC/NFD/NFKC/NFKD/`None`, a non-integer `max_decode_length` | `fix_text_batch` on Arrow input raises `PyroidError` for those options instead of delegating |
| `pyroid.fasttext` | never | — |
| `pyroid.filters` | never | unsupported parameters raise (e.g. `exclusion_writer`, a language without an English-tokenizer mapping raises `NotImplementedError`) |
| `pyroid.dedup` *(preview)* | never | — |
| `pyroid.translit` *(preview)* | never | — |

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

Other warnings are about results, not code paths, and stay plain `RuntimeWarning`s: for example,
a Python whose `unicodedata` is newer than the Unicode tables pyroid ships (17.0) gets a one-time
`RuntimeWarning`, and pyroid uses its 17.0 tables. Any other warning matches the incumbent's own
(for example `pyroid.translit.unidecode`'s lone-surrogate warning).
