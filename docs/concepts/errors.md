# Errors

pyroid's own exceptions all derive from
[`pyroid.PyroidError`](../reference/pyroid.md#pyroid.PyroidError). Where a module replaces an
incumbent, it raises the incumbent's exception types for the same inputs, so existing `except`
clauses keep working; each guide lists the exceptions.

## The hierarchy

| Exception | Bases | Raised when |
|---|---|---|
| `pyroid.PyroidError` | `Exception` | base class; also raised directly when a fallback needs a library that is not installed |
| `pyroid.InputError` | `PyroidError`, `TypeError` | an argument has the wrong type or shape: a non-`str` item (the message names its index), `bytes`, a non-string or dictionary-encoded Arrow column, a struct column or table, a bad `threads=` type |
| `pyroid.fasttext.ModelError` | `PyroidError`, `ValueError` | a model file is malformed, truncated, or not a fastText model |
| `pyroid.translit.UnidecodeError` *(preview)* | `PyroidError`, `ValueError` | `unidecode(..., errors="strict")` met a character without a transliteration; has an `index` attribute, like Unidecode's |

Because `InputError` is a `TypeError`, code that catches `TypeError` for bad arguments keeps
working.

## Standard exceptions

| Exception | When |
|---|---|
| `ValueError` | a bad value: a negative count, `threads <= 0`, a malformed Arrow buffer, an out-of-range configuration value |
| `UnicodeEncodeError` | a `str` contains a lone surrogate, for functions that cannot process one ([details](inputs.md#lone-surrogates)); a note names the document |
| `UnicodeDecodeError` | an Arrow row is not valid UTF-8; a note names the chunk and row |
| `OverflowError` | a single document over 2 GiB in a `string` (not `large_string`) Arrow output |

PyO3 adds a note `while processing '<argument>'` to errors raised while converting arguments.
For a batch, the first failing document raises, and a note on the exception names it.

## Panics

A bug in pyroid's Rust code surfaces as `pyo3_runtime.PanicException`, never as a crash of the
interpreter. It derives from `BaseException`, so a bare `except Exception` does not catch it. It
always means a pyroid bug; please
[report it](https://github.com/ao/pyroid/issues/new/choose) with the input that triggered it.
