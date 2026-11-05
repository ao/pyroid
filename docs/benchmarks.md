# Benchmarks

!!! warning "Docker/OrbStack VM on Apple Silicon, indicative, preliminary"
    Every number on this page was measured in a Linux (aarch64) Docker VM under OrbStack on an
    Apple M3 Pro host with 12 vCPUs, **while other jobs were running on the same machine**. They
    show orders of magnitude, not precise ratios: two runs of the same ftfy benchmark measured
    5.8 and 14.5 µs/doc. A run on a quiet bare-metal Linux machine is pending and will replace
    them.

## Method

The benchmark harness lives in [`bench/`](https://github.com/ao/pyroid/tree/main/bench) and
follows a few rules, enforced in code:

1. **Same output before any timing.** Every timed row first passes an equivalence gate against
   pinned reference outputs: ftfy byte-identical; language ID with the same top-k labels and
   |Δp| ≤ 1e-6; filters with the same keep/drop, reason and rewritten text; MinHash with
   identical signatures. A mismatch aborts the run.
2. **Pinned data.** FineWeb slices of 10k and 100k documents, 8 FineWeb-2 languages, `lid.176.bin`
   and `lid.176.ftz`, and a seeded adversarial set of 3,232 documents, each pinned by sha256 in
   `bench/manifest.json`.
3. **Single-process rows** use [pyperf](https://pyperf.readthedocs.io/): one value is one call over
   1,000 documents, reported as the median [interquartile range] per document. **Multi-process
   rows** use [hyperfine](https://github.com/sharkdp/hyperfine) 2.0 over datatrove's own
   `LocalPipelineExecutor(tasks=8, workers=8)`, pinned to 8 CPUs, 3 runs after a warm-up.
4. **CPU-seconds and peak memory** are recorded for the harness rows.
5. Charts are horizontal bars on a linear axis, fastest on top, with incumbents and close rivals
   named.

The fastText rows below are the harness's own, gated rows. The other pyroid numbers come from
each module's benchmark scripts: the filters and translit scripts check that pyroid's output
equals the incumbent's on the timed documents before timing, and the ftfy and dedup timings rest
on their separate [parity runs](compatibility.md). pyroid's rows in the shared harness for the
other modules are still being added (see [What's pending](#whats-pending)).

To reproduce, from a checkout on Linux (or in a Linux container), with a pyroid wheel built
into `dist/`:

```console
$ cd bench
$ uv run --frozen python fetch.py                     # the pinned inputs
$ BENCH_IMPL="ftfy=pyroid.ftfy fasttext=pyroid.fasttext" \
    uv run --frozen --with ../dist/pyroid-*.whl bash run_baselines.sh
```

`bench/README.md` documents every option, including `BENCH_RIGOROUS=1` for a quiet machine.

## Where the CPU goes

datatrove 0.10.1's FineWeb-style chain (ftfy → language ID → Gopher repetition → Gopher quality →
C4 → FineWeb quality → MinHash) on the first 10k FineWeb documents (30.5 M characters),
CPython 3.13, one process. Share of chain = the stage's single-process time over the sum of all
stages.

| Stage | Incumbent | ms/doc, 1 process | Share of chain |
|---|---|---:|---:|
| fix_text | ftfy 6.3.1 | 0.88 [0.84–0.94] | 2.3% |
| language ID | datatrove `LanguageFilter` (fastText lid.176.bin) | 0.39 [0.36–0.44] | 1.0% |
| filter | `GopherRepetitionFilter` | 8.65 [8.47–9.20] | 22.2% |
| filter | `GopherQualityFilter` | 7.21 [6.72–7.68] | 18.5% |
| filter | `C4QualityFilter` (FineWeb settings) | 8.42 [7.81–8.87] | 21.6% |
| filter | `FineWebQualityFilter` | 6.54 [6.23–6.99] | 16.8% |
| MinHash | datatrove `MinhashDedupSignature` | 6.88 [6.72–7.07] | 17.7% |
| **whole chain** | **datatrove, 1 process** | **39.9** | 100% |

The four filters are about **79%** of the CPU and MinHash about **18%**; ftfy and language ID
together are about 3%. Speeding up `fix_text` 10× on its own makes this pipeline about 1.02×
faster. datatrove tokenizes English with spaCy's `blank("en")` tokenizer for both the filters
and MinHash; in MinHash, spaCy is about 70% of the cost.

At 8 processes the same chain took 63.2 s [60.6–66.1] wall time (464.5 CPU-s) for 10k documents,
6.3× faster than one process, with a peak of 2.36 GiB of anonymous memory.

## `pyroid.ftfy`

`fix_text` per document, single-threaded, best of 3, the same documents for both:

| Corpus | ftfy 6.3.1 (µs/doc) | pyroid.ftfy (µs/doc) | Speedup |
|---|---:|---:|---:|
| FineWeb, first 10k docs, CPython 3.15 | 693.0 | 7.3 | 95× |
| FineWeb, first 10k docs, CPython 3.12 | 969.3 | 5.8 | 167× |
| FineWeb-2 Spanish / German / French (250 docs each), 3.15 | 1,213 / 810 / 1,988 | 12.7 / 9.3 / 28.9 | 95× / 87× / 69× |
| FineWeb-2 Chinese / Russian / Japanese / Arabic, 3.15 | 960 / 1,199 / 701 / 900 | 33.5 / 40.7 / 38.0 / 39.2 | 29× / 29× / 18× / 23× |
| FineWeb-2 Hindi, 3.15 (slowest) | 799 | 56.1 | 14× |
| Adversarial set (3,232 mojibake-heavy docs) | 970 | 30.3 | 32× |

An identical earlier run of the FineWeb rows gave 71× (3.12) and 93× (3.15). The harness baseline
for ftfy is 0.88 ms/doc.

## `pyroid.fasttext`

Harness `stages.py` language-ID rows (equivalence-gated; median ms/doc; 1,000 FineWeb documents,
newlines replaced by spaces), against fasttext-predict 0.9.2.4:

| Model | fasttext-predict | pyroid | Speedup |
|---|---:|---:|---:|
| lid.176.bin, k=1 | 0.389 | 0.176 | 2.2× |
| lid.176.bin, k=5 | 0.377 | 0.248 | 1.5× |
| lid.176.ftz, k=1 | 0.650 | 0.121 | 5.4× |
| lid.176.ftz, k=5 | 0.636 | 0.139 | 4.6× |

`load_model` in a fresh process: 50–60 ms (`.bin`) and 4–7 ms (`.ftz`) for pyroid, against
115–205 ms and 38–68 ms.

**Threads in one process** (free-threaded CPython 3.15t, 10k documents, best of 5, ms):

| Model | 1 | 2 | 4 | 8 | 8 threads vs single-threaded fastText |
|---|---:|---:|---:|---:|---:|
| .bin, `predict(batch, threads=n)` | 2,061 | 839 | 384 | 247 | 13.9× |
| .bin, n Python threads × `predict(str)` | 2,436 | 1,143 | 636 | 401 | 8.5× |
| .ftz, `predict(batch, threads=n)` | 1,517 | 859 | 632 | 316 | 20× |

Single-threaded fastText (fasttext-predict, CPython 3.13) took 3,426 ms (`.bin`) and 6,354 ms
(`.ftz`) for the same 10k documents.

**Memory:** with eight threads in one process, the model and predictions added 132 MiB (`.bin`) or
1 MiB (`.ftz`), against about 1.04 GiB for eight per-process copies of the model. The whole
process peaked at 473 MiB, 320 MiB of which was the corpus itself, about 25% of the 1.87 GiB that
datatrove's 8-process language-ID stage holds.

## `pyroid.filters`

One core, CPython 3.13, the first 2,000 FineWeb documents, median of 5 (datatrove: median of 2);
every result was checked equal to datatrove's before timing. µs/doc:

| Filter | datatrove | pyroid, list | pyroid, Arrow | Speedup |
|---|---:|---:|---:|---:|
| GopherRepetition | 8,984 | 185 | 217 | 48× |
| GopherQuality | 7,160 | 101 | 89 | 71× |
| C4Quality (default) | 7,654 | 253 | 192 | 30× |
| C4Quality (FineWeb settings) | 10,585 | 204 | 215 | 52× |
| FineWebQuality | 8,875 | 85 | 89 | 105× |
| FineWeb's 4 filters, chained (`apply`) | 23,875 | 277 | 363 | **86×** |

datatrove's chain costs less than the sum of its filters because it caches tokenization between
them; `apply` shares it too. A separate re-check of three separate filter calls measured 0.43 ms
per document on one thread, about 52× datatrove, and 0.097 ms per document at 8 threads.

## `pyroid.dedup` *(preview)*

Signatures from raw text (normalization, spaCy-identical tokenization and hashing all included),
single thread, 1,000 FineWeb documents, CPython 3.13, median [IQR] or the range over two 15-run
sets:

| Implementation | ms/doc | Same decisions as datatrove? |
|---|---:|---|
| datatrove `MinhashDedupSignature`, sha1/64 (3 runs) | 6.17 [6.13–6.30] (harness: 6.88) | — |
| datasketch 2.0.0 `MinHash.bulk` | 0.551 | no (different shingling) |
| rensa 0.5.0, with the harness's Python shingling | 0.168–0.173 | no (different shingling) |
| **pyroid `minhash_signatures`, sha1/64** | **0.094–0.099** | yes |
| pyroid, xxhash/64 | 0.085–0.091 | yes |

That is about 64× datatrove in the same session (71× against the harness baseline) and 1.7–1.8×
rensa, with datatrove's decisions. One noisier 7-run set gave pyroid 0.144 [0.108–0.164]. In
datatrove, spaCy is about 70% of this cost.

The whole pipeline, `MinHashDedup.fit` (sha1/64, free-threaded CPython 3.15t, median of 3 fresh
processes, wall seconds):

| Documents | 1 thread | 8 threads | datatrove's 4 stages |
|---|---:|---:|---:|
| 10k | 1.21 | 0.29 | 99.5 (1 process); 13.78 (8 processes) |
| 100k | 12.86 | 2.36 | — |

The 1-process datatrove figure was measured earlier, under heavier load.

For exact duplicates (100k documents, CPython 3.15t), Python's own `set` over a `list[str]` beat
`pyroid.dedup.exact_dedup`: 0.020 s against 0.071 s on one thread. `exact_dedup` is for Arrow
input, normalized comparison and streaming state.

## `pyroid.translit` *(preview)*

From the translit module's working notes (final numbers pending). Single thread, CPython 3.12,
fresh `str` objects, best of ≥5 runs, µs/doc:

| Corpus | anyascii (Python) | pyroid `anyascii` | Unidecode 1.4.0 | pyroid `unidecode` |
|---|---:|---:|---:|---:|
| FineWeb, first 10k docs | 80.77 | 2.19 (37×) | 192.00 | 2.23 (86×) |
| FineWeb-2 Arabic (250 docs) | 283.81 | 23.17 (12×) | 486.91 | 23.22 (21×) |
| FineWeb-2 Chinese (250 docs) | 149.28 | 9.82 (15×) | 253.08 | 9.53 (27×) |

`pyroid.translit.unidecode`'s output is not Unidecode's; see
[how different it is](guides/translit.md#how-different-is-it-from-unidecode).

## Thread scaling

Speedup at 8 threads over 1 thread, one process, free-threaded CPython 3.15t (GIL off):

| Module and call | 8 threads vs 1 |
|---|---:|
| `ftfy.fix_text_batch`, list / Arrow (100k docs) | 2.4–3.6× / 2.1–3.1× |
| `ftfy.fix_text` from 8 Python threads | 3.2–5.8× |
| `fasttext` `.bin` / `.ftz`, `threads=8` | 8.3× / 4.8× (`.ftz` under load; needs a quiet rerun) |
| `filters.apply`, FineWeb's 4 filters, list / Arrow | 4.3× / 5.0× |
| `dedup.MinHashDedup.fit`, 10k / 100k docs (preview) | 4.2× / 5.4× |
| `translit.anyascii`, FineWeb 100k, list / Arrow (preview, working notes) | 1.27× / 4.58× |

In the ftfy measurements scaling was limited by the memory allocator, not the kernels: with
glibc's allocator tuned to keep freed memory, `fix_text_batch` on Arrow went from 5.5 µs/doc on
one thread to 1.5 µs/doc on eight (3.6×), against 2.7–3.5 µs/doc at eight threads untuned. A
different allocator is being evaluated. translit's list rows use fresh `str` objects, whose UTF-8
form CPython builds on first access; with it already cached, list input scaled 1.73×.

## What's pending

- A **bare-metal Linux run** of everything on this page, on a quiet machine, with the harness's
  rigorous mode. Its numbers will replace these.
- **pyroid in the end-to-end pipeline**: the harness can swap each stage of datatrove's pipeline
  for pyroid's and check the output digest. That run hasn't been done, so there is no end-to-end
  pyroid number yet.
- pyroid rows in the harness for filters, dedup and translit; x86-64 numbers.
