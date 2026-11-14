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

The pool is created on first use, not at import. Its size comes from the `PYROID_NUM_THREADS`
environment variable, read once at that point. If it is unset, rayon's default applies
(`RAYON_NUM_THREADS`, else the number of available CPUs).

Results never depend on `threads`: the same input gives the same output for every value.

Each task on the pool holds at least 64 documents, because handing out a task costs more than a
few documents' work. So a batch of `n` documents uses at most `n / 64` threads: fewer than 128
documents run on the calling thread whatever `threads` says, 128 documents use two threads, and
eight threads need 512 documents or more. To spread a smaller batch of slow documents over
cores, call pyroid from several Python threads instead, on a free-threaded build. Worker threads do not survive `fork()`, so the pool is rebuilt
in a forked child process.

## The GIL

While a batch runs, pyroid releases the GIL (the Rust kernels never call into Python). Other
Python threads keep running, and a pyroid call in one thread doesn't block another thread's
pyroid call. Converting the results back to Python objects needs the interpreter, so it happens
on the calling thread after the batch.

## Free-threaded Python

pyroid ships `cp314t` and `abi3t` wheels for free-threaded CPython 3.14t and 3.15t, and declares
itself safe to run without the GIL. Importing it does not re-enable the GIL
(`sys._is_gil_enabled()` stays `False`). The test suite runs under
[pytest-run-parallel](https://github.com/Quansight-Labs/pytest-run-parallel) on both.

Loaded objects are immutable and meant to be shared:

- a `pyroid.fasttext.Model` is read-only, so load it once and call it from any number of threads;
- filter configurations are built once per configuration and cached.

On a free-threaded build you can therefore choose between one call with `threads=n` and `n`
Python threads each calling pyroid on their own slice. Which is faster depends on the module.
The modules' earlier runs, in a Docker VM (indicative), measured this:

- `pyroid.fasttext`, 10k documents: `predict(batch, threads=8)` took 247 ms, and eight Python
  threads calling `predict(str)` took 401 ms.
- `pyroid.ftfy`: eight Python threads calling `fix_text` scaled 3.2–5.8× over one thread.

For one call with a pool, on an AWS EC2 c7i.4xlarge, `fix_text_batch(threads=8)` on 10k FineWeb
documents took 2.06 µs per document against 6.70 µs on one thread with list input, and 1.31
against 6.63 µs with Arrow input ([Benchmarks](../benchmarks.md#thread-scaling)).

!!! note "Arrow input scales best"
    On the same c7i, 8 threads took `translit.anyascii` from 1.79 to 0.60 µs per document on
    Arrow input, but only from 1.74 to 1.01 µs on a list. For a list, pyroid still builds the
    output strings one after another on the calling thread. See
    [Benchmarks](../benchmarks.md#thread-scaling).

## What can bring the GIL back

pyroid itself never re-enables the GIL, but a library it hands work to can:

- `import sacrebleu` loads lxml, which re-enables the GIL on free-threaded builds.
  `pyroid.metrics.sacrebleu` never imports sacrebleu, so it stays GIL-free unless a call is
  [delegated](fallbacks.md) to the real sacrebleu.
- `pyroid.trafilatura.extract` and `extract_batch` never import trafilatura or lxml. A delegated
  option does, and so does `bare_extraction` when trafilatura is installed (its `Document`
  subclasses trafilatura's) or when its `body` is read as an lxml element; lxml re-enables the
  GIL.
- With `python_tokenizer=True`, the filters and MinHash call datatrove's own tokenizer in
  Python on the calling thread, so that part runs at the tokenizer's speed and doesn't scale with
  `threads=` ([Languages](../guides/languages.md)).

`pyroid.patch("fasttext")` also lets code written for fastText run on 3.15t, where fastText has
no wheel. For example, fast-langdetect works patched on 3.15t, where its unpatched import fails
([`pyroid.patch()`](../guides/patch.md)).
