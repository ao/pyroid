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
environment variable, read once at that point. If it is unset or empty, rayon's default applies
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

A call on a single `str` releases the GIL only for long documents (4 KiB and more) or slow
kernels (fastText, the filters, MinHash). When another Python thread is busy, getting the GIL
back after such a call can take up to the interpreter's switch interval
(`sys.getswitchinterval()`, 5 ms by default), which is far longer than the call itself. So with
busy threads around, one call per document can be many times slower than alone; a batch call
releases and retakes the GIL once for the whole batch.

## Ctrl-C

A batch runs without the GIL, so Python cannot run its signal handlers while it works. pyroid runs
them itself: every 100 ms the thread that made the call runs the pending handlers, and if one
raises (`KeyboardInterrupt` for Ctrl-C), the batch stops at its next document and the call raises
that exception. The result so far is thrown away, and objects such as a `MinHashLSH` or a
`MinHashDedup` keep their state from before the call. Two keep what was done: a trafilatura
`LRUCache` holds the pages extracted so far, as after an interrupted loop of `extract` calls, and
a MinHash signature writer interrupted in `finish()` has to be started again. A handler that
returns lets the batch go on.
Only the main thread runs signal handlers, as in Python, so a batch on another thread runs to its
end. A single document is not interrupted halfway, so a very long one finishes first.

## Memory after a call

Each pool thread keeps a few buffers between calls so that a batch allocates little per
document: the spaCy tokenizer's cache of up to 16,384 word spans per language (about 2 MiB when
full), and scratch buffers of up to 1 MiB each, which are given back when a long document made
them bigger. On a machine with many cores the pool has as many threads, and their buffers add up:
after `gopher_quality` on 100k FineWeb documents with 64 threads, RSS stayed about 240 MiB above
where it was with Arrow input (EC2 c8g.16xlarge, shared, indicative). Two more things keep memory
after a call:

- A list of `str` that is not ASCII gets a UTF-8 copy of each string, which CPython keeps inside
  the string object until the string is freed (pyroid reads text as UTF-8). That was another
  240 MiB for those 100k documents. Arrow input has no such copy.
- The C allocator keeps freed memory per thread for reuse. With glibc, `MALLOC_ARENA_MAX=2` in the
  environment, or `ctypes.CDLL("libc.so.6").malloc_trim(0)` after a large call, gives more of it
  back to the system.

`threads=` caps how many pool threads a call uses, and `PYROID_NUM_THREADS` the size of the pool.

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
