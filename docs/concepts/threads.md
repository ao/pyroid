# Threads and free-threading

pyroid's batch functions run on a shared Rust thread pool, without holding the GIL. You get
parallelism from one call on an ordinary CPython, and on free-threaded CPython (3.14t, 3.15t)
you can also call pyroid from many Python threads at once.

## `threads=`

Every batch function takes a keyword-only `threads=` argument:

| Value | Meaning |
|---|---|
| `None` (default) | use the whole shared pool |
| `n` | run at most `n` tasks at once on the pool |
| `1` | run on the calling thread, no pool |

```python
import pyroid.ftfy as ftfy

ftfy.fix_text_batch(docs)  # all cores
ftfy.fix_text_batch(docs, threads=4)  # at most 4 at once
ftfy.fix_text_batch(docs, threads=1)  # this thread only
```

The pool is created on first use (not at import), sized by the `PYROID_NUM_THREADS` environment
variable, which is read once at that point; if it is unset, rayon's default applies
(`RAYON_NUM_THREADS`, else the number of available CPUs). Results never depend on `threads`: the same input gives the same output
for every value. Batches of 64 documents or fewer run on the calling thread, where the pool's
overhead would cost more than it saves. Worker threads do not survive `fork()`, so the pool is
rebuilt in a forked child process.

## The GIL

While a batch runs, pyroid releases the GIL (the Rust kernels never call into Python). Other
Python threads keep running, and a pyroid call in one thread doesn't block another thread's
pyroid call. Converting the results back to Python objects needs the interpreter, so it happens
on the calling thread after the batch.

## Free-threaded Python

pyroid ships `cp314t` and `abi3t` wheels for free-threaded CPython 3.14t and 3.15t, and declares
itself safe to run without the GIL: importing it does not re-enable the GIL
(`sys._is_gil_enabled()` stays `False`). The test suite runs under
[pytest-run-parallel](https://github.com/Quansight-Labs/pytest-run-parallel) on both.

Loaded objects are immutable and meant to be shared:

- a `pyroid.fasttext.Model` is read-only, so load it once and call it from any number of threads;
- filter configurations are built once per configuration and cached.

On a free-threaded build you can therefore choose between one call with `threads=n` and `n`
Python threads each calling pyroid on their own slice. Which is faster depends on the module so
far (Docker VM, indicative; [Benchmarks](../benchmarks.md#thread-scaling)):

- `pyroid.fasttext`, 10k documents: `predict(batch, threads=8)` took 247 ms, eight Python
  threads calling `predict(str)` 401 ms (the module's earlier run).
- `pyroid.ftfy`: in the module's earlier run, eight Python threads calling `fix_text` scaled
  3.2–5.8× over one thread. In the consolidated run, `fix_text_batch(threads=8)` on 10k FineWeb
  documents took 1.72 µs per document against 5.66 µs on one thread (list input), and 1.08
  against 5.16 µs on Arrow input.

!!! note "Arrow input scales best"
    In the consolidated run (Docker VM, indicative), 8 threads took `translit.anyascii` from
    1.15 to 0.41 µs per document on Arrow input, but from 1.62 to 1.00 µs on a list. For a list,
    pyroid still builds the output strings one after another on the calling thread. See
    [Benchmarks](../benchmarks.md#thread-scaling).
