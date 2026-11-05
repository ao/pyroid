# Writing a pyroid module

Every product module (`ftfy`, `fasttext`, `filters`, `dedup`, `translit`) follows the
same shape as the demonstration module `pyroid.text`. A track touches **only its own
files plus one registration line per shared file** (two in total), so tracks merge
without conflicts.

## Layout

| Path | What |
|---|---|
| `crates/pyroid-core/src/<mod>/` | Pure-Rust kernels (`&str -> ...`). No PyO3, `#![forbid(unsafe_code)]`, unit tests + doctests. Add `pub mod <mod>;` to `crates/pyroid-core/src/lib.rs` (**registration line 1**). |
| `crates/pyroid-py/src/<mod>.rs` | `#[pyfunction]` wrappers and `pub fn register(parent)`, which creates the native submodule `pyroid.<mod>`. Add `mod <mod>;` and `<mod>::register(m)?;` to `crates/pyroid-py/src/lib.rs` (**registration line 2**). |
| `python/pyroid/<mod>.py` | Public module: re-exports the native functions (`from pyroid._pyroid import <mod> as _native`). Pure-Python glue only where it adds value. |
| `python/pyroid/<mod>.pyi` | Hand-written stub: `@overload`s per input kind, Google-style docstrings (the API docs are built from these). |
| `python/tests/test_<mod>*.py` | Tests against the installed wheel. |

`python/pyroid/__init__.py` does not import submodules; users write `import pyroid.text`.
New exception types go in `crates/pyroid-py/src/errors.rs` (subclass `PyroidError`) and are
re-exported from `__init__.py`/`_pyroid.pyi`; ask the core owner first.

## API conventions (what users see)

```python
import pyarrow as pa, pyroid.text as t
t.normalize_whitespace("  a \t b ")            # 'a b'                 str  -> str
t.normalize_whitespace(["a  b", "ok"])          # ['a b', 'ok']         list -> list ('ok' is the same object)
t.min_chars(("abc", "x"), 2)                    # (True, False)         tuple -> tuple
col = t.normalize_whitespace(pa.array(["a  b", None]))   # pyroid.Column<string>
pa.chunked_array(col); pa.array(col); polars.Series(col)  # zero-copy into any Arrow library
t.min_chars(pa.chunked_array([["abc"], [None]]), 2).to_pylist()   # [True, None]
t.normalize_whitespace(docs, threads=4)         # at most 4 tasks at once on the shared pool
```

- **Input** (`TextInput`): `str`; `list[str]`/`tuple[str, ...]`; or any object with
  `__arrow_c_stream__` (preferred) or `__arrow_c_array__` whose type is `string`,
  `large_string` or `string_view` (pyarrow, polars, pandas 3, HF datasets, nanoarrow,
  arro3, duckdb, `pyroid.Column`). Strings are never copied on the way in.
- **Transforms** return the caller's kind: str → str, list → list, tuple → tuple,
  Arrow → `pyroid.Column` (`large_string` for `large_string` input, else `string`).
  A document the kernel leaves unchanged comes back as **the original object**.
- **Predicates** return a mask in the caller's kind: `bool`, `list[bool]`,
  `tuple[bool, ...]`, or a `Column` of `bool`. They never rebuild strings.
- **Arrow nulls** stay null (transform → null string, predicate → null bool).
  `None` inside a list is an error.
- **`threads=`** (keyword-only, default `None`): `None` = the whole shared pool
  (sized by `PYROID_NUM_THREADS`, else rayon's default); `n` = at most `n` tasks at once;
  `1` = run on the calling thread. The pool is built on first use and rebuilt in a
  forked child. Batches of ≤ 64 docs always run on the calling thread.
- **`pyroid.Column`**: zero-dependency result type; `len()`, `.type`, `.name` (carried over
  from the input field), `.null_count`, `.num_chunks`, `.to_pylist()`,
  `__arrow_c_stream__`, `__arrow_c_array__` (concatenates chunks). pyarrow is never
  required at runtime. polars 2.1 takes the Series name from the stream and ignores
  `pl.Series("new", col)`'s name: use `pl.Series(col).alias("new")`.
- **Errors**: `pyroid.InputError(PyroidError, TypeError)` for a wrong type anywhere
  (non-str input or item — the message names the index —, bytes, non-string Arrow type,
  dictionary-encoded or struct/table columns, bad `threads`/count types).
  `ValueError` for bad values (negative `n`, `threads <= 0`, malformed Arrow buffers).
  `UnicodeEncodeError` (with a note naming the item) for lone surrogates in a str;
  `UnicodeDecodeError` (with a note naming chunk and row) for invalid UTF-8 in Arrow.
  A `string` output chunk that would pass 2 GiB is split into several chunks;
  `OverflowError` only for a single > 2 GiB document, or `__arrow_c_array__` concatenation
  past 2 GiB (use `pa.chunked_array`). PyO3 adds the note
  `while processing '<arg>'`. A Rust panic surfaces as `pyo3_runtime.PanicException`
  (a `BaseException`: it means a pyroid bug), never an abort.
- **Fallbacks**: when a module hands a call to the incumbent library because the Rust port
  does not handle it (an option value it does not support, a helper it does not port), it
  returns the incumbent's result and emits **`pyroid.FallbackWarning`** (a `RuntimeWarning`):
  that class exactly, never a bare `RuntimeWarning` or a subclass, once per process for each
  function and reason. If the incumbent is not installed, raise `PyroidError` instead.
  Warnings about results (an incumbent at another version, missing Unicode tables) stay
  plain `RuntimeWarning`s. Users silence the slow-path notice with
  `warnings.simplefilter("ignore", pyroid.FallbackWarning)`; the parity suites ignore it,
  because it says which code path ran, not what the result is. Any other warning must match
  the incumbent's.

## Rust side

```rust
// crates/pyroid-py/src/<mod>.rs
use pyo3::prelude::*;
use crate::batch::{predicate, transform};
use crate::input::{Count, TextInput, Threads};

/// Docstring (becomes __doc__; keep it in sync with the .pyi).
#[pyfunction]
#[pyo3(signature = (x, /, *, threads = None))]
fn fix_text(py: Python<'_>, x: TextInput, threads: Option<Threads>) -> PyResult<Bound<'_, PyAny>> {
    transform(py, x, threads, pyroid_core::ftfy::fix_text)   // for<'a> Fn(&'a str) -> Cow<'a, str> + Sync
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.ftfy")?;   // "pyroid.<mod>": __module__ of the functions
    m.gil_used(false)?;
    m.add_function(wrap_pyfunction!(fix_text, &m)?)?;
    parent.add_submodule(&m)
}
```

- `batch::transform(py, x, threads, f)` with `f: for<'a> Fn(&'a str) -> Cow<'a, str>`.
  Return `Cow::Borrowed(s)` of the **whole** input when nothing changes (a borrowed
  substring counts as a change and is copied).
- `batch::predicate(py, x, threads, f)` with `f: Fn(&str) -> bool`.
- `batch::map_values(py, &x, threads, f) -> Vec<Option<R>>` for anything else (scores,
  hashes, labels); build the output yourself (e.g. `PyList::new` from an exact-size
  iterator, a numpy array via `PyArray1::from_vec`).
- Kernels are **pure Rust**: they run detached on rayon workers and must never call
  Python (clippy's `disallowed-methods` bans `Python::attach`). Anything that needs the
  interpreter happens before or after the batch, on the calling thread.
- Build regexes, tables and models once (`std::sync::LazyLock`, or a `#[pyclass(frozen)]`
  that owns them), never per document. Wrap model state in a `#[pyclass(frozen)]` and
  pass `&self` data into the closure; it must be `Sync`.
- `unsafe` is only allowed in `crates/pyroid-py/src/arrow/`. Every block has a
  `// SAFETY:` comment and one unsafe operation (clippy enforces both).
- Arrow trust model: as in arrow-rs, producers are trusted on buffer sizes,
  schema/array consistency and `n_buffers`; everything else (shape, null counts,
  offsets, views, UTF-8, already-consumed capsules) is checked, and a mutating
  producer can't push reads out of bounds.
- Errors: return `PyResult`; use `errors::input_error(py, msg)` for wrong types.

## Tests (`python/tests/test_<mod>*.py`)

1. **Reference parity**: a Hypothesis property test against the pure-Python or incumbent
   reference (e.g. `" ".join(s.split())`, `ftfy.fix_text`), over str, list, tuple and
   Arrow. Generate surrogates explicitly (`st.characters(categories=["Cs"])`) and assert
   the documented error.
2. **Container kinds**: str/list/tuple/Arrow in, the same kind out; unchanged docs are the
   same objects (`out[i] is x[i]`); Arrow nulls, slices (`offset > 0`), `string`,
   `large_string`, `string_view`, multi-chunk inputs.
3. **Arrow without pyarrow**: `pyroid._pyroid._testing.column(values, large=..., chunks=...)`
   builds a `Column` with no Arrow library; pyarrow/polars tests use `pytest.importorskip`.
4. **Threads**: results identical for `threads=None/1/2/…`; large batches (10k+ docs)
   complete under a timeout. Mark tests that use subprocesses, `pytest.warns`, or global
   state `@pytest.mark.thread_unsafe` (pytest-run-parallel runs the suite on 3.14t/3.15t).
5. **Stubs**: stubtest runs in CI over the installed `pyroid` package (it recurses into
   `pyroid.<mod>` and `pyroid._pyroid`); mypy --strict type-checks the tests against the
   stubs. Allowlist entries go in `.stubtest-allowlist`, each with a reason.

## Checklist for a product track

- [ ] Kernels in `pyroid-core/src/<mod>/` with `cargo test` coverage; no Python, no `unsafe`.
- [ ] `pyroid-py/src/<mod>.rs` uses `transform` / `predicate` / `map_values`; no Python
      calls inside closures; per-document state hoisted out of the loop.
- [ ] Two registration lines (core `lib.rs`, bindings `lib.rs`).
- [ ] `<mod>.py` re-export + `<mod>.pyi` overloads for str / list / tuple / Arrow,
      with docstrings; stubtest clean.
- [ ] Hypothesis parity vs the reference, surrogate policy, Arrow nulls and types,
      identity of unchanged docs, threads equivalence, a 10k-doc timeout test.
- [ ] `cargo fmt`, clippy pedantic `-D warnings`, ruff, mypy --strict, cargo-deny, typos clean.
- [ ] Wheel built in Docker; pytest green on the installed wheel on 3.11, 3.12, 3.14t,
      3.15, 3.15t; `sys._is_gil_enabled()` stays False on 3.14t/3.15t.
- [ ] Every delegation to the incumbent warns once with `pyroid.FallbackWarning`.
- [ ] Speed numbers only from the harness (`bench/`), labelled with the machine.
