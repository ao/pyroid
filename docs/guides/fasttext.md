# `pyroid.fasttext`: fastText inference

`pyroid.fasttext` is a pure-Rust port of [fastText](https://github.com/facebookresearch/fastText)'s
inference (fastText is © Facebook, Inc., MIT License). It loads the same `.bin` and quantized
`.ftz` model files. Its labels and probabilities are bit-identical to fastText's C++ predict path,
as built in fasttext-predict 0.9.2.4 ([parity](../compatibility.md#fasttext)).

!!! note "Results differ slightly between platforms"
    pyroid reproduces the C++ code's floating-point operations, including which multiply-adds
    are fused, and calls the C library's `exp`, `expf` and `log` as fastText does. Both vary by
    platform, so the last bits of a probability depend on the architecture and the C library.
    On Linux (glibc) and Windows x64, pyroid matches fasttext-predict bit for bit. On macOS,
    fasttext-predict's clang build rounds its dot products differently: the largest difference
    on GitHub's runners was 1.91e-6, on a test model with weights scaled by 20. Different builds
    of fastText itself differ too: fasttext-numpy2-wheel's probabilities were up to 1.79e-7 away
    from fasttext-predict's on the same inputs.

Besides speed, there are two reasons to switch. pyroid has wheels for CPython 3.11–3.15 and
free-threaded 3.14t/3.15t on Linux (glibc and musl), macOS and Windows, so it installs without a
compiler. The upstream `fasttext` 0.9.3 is an sdist that needs a C++ compiler and breaks on
numpy 2, and the wheel-shipping forks stop short of some of those interpreters.

The other reason is threads. A loaded model is immutable and shared by all threads, and batches
run without the GIL, so one process with eight threads holds one copy of the model instead of
eight.

```console
$ pip install "pyroid[fasttext]"     # pyroid.fasttext returns numpy arrays
```

## Change one import

```diff
-import fasttext
+import pyroid.fasttext as fasttext

 model = fasttext.load_model("lid.176.bin")
 model.predict("Bonjour tout le monde", k=2)
 # (('__label__fr', '__label__en'), array([9.94570732e-01, 8.63876310e-04]))
```

`lid.176.bin` (125 MiB) and the compressed `lid.176.ftz` (916 KiB) are fastText's language-ID
models for 176 languages, published on the
[fastText site](https://fasttext.cc/docs/en/language-identification.html) (CC BY-SA 3.0):

```console
$ curl -LO https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
$ curl -LO https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
```

`pyroid clean` and `pyroid langid` download `lid.176.bin` themselves on first use
([details](cli.md#the-language-id-model)).

The inference methods of fastText's `_FastText` are there, with the same results:
`predict`, `get_sentence_vector`, `get_word_vector`, `get_input_vector`, `get_words`,
`get_labels`, `words`, `labels`, `get_line`, `get_subwords`, `get_word_id`, `get_label_id`,
`get_subword_id`, `get_dimension`, `is_quantized`, `get_input_matrix`, `get_output_matrix`,
`model[word]`, `word in model`, and the module-level `tokenize`.

Training, `test`, `quantize`, `save_model`, `get_nearest_neighbors` and `get_analogies` are out
of scope. Train with fastText, then load the file with pyroid.

## Batches

`predict` and `get_sentence_vector` also take a batch: a list or tuple of `str`, or an Arrow string
column. A batch runs on pyroid's thread pool without the GIL.

```python
import pyroid.fasttext as fasttext

model = fasttext.load_model("lid.176.ftz")
lines = [d.replace("\n", " ") for d in docs]  # one line per document, as with fastText

labels, probs = model.predict(lines, k=1)  # list of label lists, list of float32 arrays
labels, probs = model.predict(lines, k=1, threads=8)
vectors = model.get_sentence_vector(lines)  # 2-D float32 array, (len(lines), dim)
```

| Input | `predict` returns | `get_sentence_vector` returns |
|---|---|---|
| `str` | `(tuple[str, ...], float64 array)`, as fastText | 1-D float32 array |
| list, tuple or Arrow | `(list[list[str]], list[float32 array])`, one entry per line; `None` entries for Arrow nulls | 2-D float32 array; NaN rows for Arrow nulls |

As in fastText, a line must not contain `"\n"` (`ValueError`), so replace newlines first.

### Language ID over a Parquet column

```python
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyroid.fasttext as fasttext

model = fasttext.load_model("lid.176.bin")
text = pq.read_table("fineweb.parquet", columns=["text"])["text"]
labels, probs = model.predict(pc.replace_substring(text, "\n", " "), k=1)
lang = [l[0].removeprefix("__label__") if l else None for l in labels]
```

## Free-threaded Python

On 3.14t and 3.15t you can also call one shared model from many Python threads:

```python
from concurrent.futures import ThreadPoolExecutor

model = fasttext.load_model("lid.176.bin")  # load once, share
with ThreadPoolExecutor(8) as pool:
    results = list(pool.map(lambda line: model.predict(line, k=1), lines))
```

A single batched call with `threads=8` was faster than eight Python threads in the measurements
so far ([Threads](../concepts/threads.md)).

## Differences from fastText

These are deliberate, or harmless for real models (from the module's tests and documentation):

- A missing or unreadable file raises `OSError` (`FileNotFoundError`, `IsADirectoryError`, ...)
  where fastText raises `ValueError`. A truncated or corrupt file raises
  [`ModelError`](../reference/fasttext.md#pyroid.fasttext.ModelError) (a `ValueError`) where
  fastText may crash, hang or read garbage.
- pyroid has limits that fastText doesn't, which only reject crafted files: `maxn` must be 0–32,
  `wordNgrams` at most 32, and dictionary probe runs at most 1024 (lid.176.bin's longest is 99).
- `get_input_vector` out of range raises `IndexError` (undefined behaviour in fastText). Tuples
  and Arrow columns are accepted. A non-`str` item raises `pyroid.InputError` (a `TypeError`)
  where fastText raises `AttributeError`.
- A lone surrogate gives a `TypeError` with pyroid's own message. The type is the same, and a
  `"\n"` still takes precedence, as in fastText.
- A `k` above 2^63 raises `OverflowError` (fastText: `TypeError`). Outside the int32 range it is
  `TypeError`, as in fastText.
- Tie order follows libstdc++, as fastText's Linux builds do. Builds against libc++ or MSVC's
  standard library may order exact ties differently.
- `get_subwords` returns int64 ids even when empty (fastText gives float64 when every n-gram is
  pruned).
- The return types are the `fasttext` package's, not fasttext-predict's, which differ:
  fasttext-predict 0.9.2.4's `predict(text)` gives the probabilities as a tuple of floats (pyroid
  and `fasttext`: a float64 array), and its `predict(lines)` returns only the labels, so
  `labels, probs = model.predict(lines)` fails there unless there are exactly two lines. pyroid
  gives `(labels, probabilities)` for every line, as `fasttext` does. fasttext-predict has no
  `get_sentence_vector`.

pyroid never delegates to fastText, so it never emits `FallbackWarning`.

## Speed and memory

On an AWS c7i.4xlarge (Intel Xeon Platinum 8488C), over the first 10,000 FineWeb documents at
`k=1`, in µs per document:

| Model | fasttext-predict 0.9.2.4 | pyroid, 1 thread | pyroid, 8 threads | Speedup, 1 thread |
|---|---:|---:|---:|---:|
| `lid.176.bin` | 185 | 114 | 16.8 | 1.6× |
| `lid.176.ftz` | 373 | 103 | 16.3 | 3.6× |

pyroid ran on free-threaded CPython 3.15t and fasttext-predict on CPython 3.13. With eight
threads, one process holds one model copy instead of eight. On a c8g.4xlarge (AWS Graviton4) the
speedups were 2.4× and 5.6×. See [Benchmarks](../benchmarks.md#per-stage).

`k=5`, load times and memory are in the module's earlier, preliminary run (Docker/OrbStack VM on
Apple Silicon, indicative; [details](../benchmarks.md#pyroidfasttext)).

API: [`pyroid.fasttext` reference](../reference/fasttext.md).
