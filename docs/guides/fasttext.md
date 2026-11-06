# `pyroid.fasttext`: fastText inference

`pyroid.fasttext` is a pure-Rust port of [fastText](https://github.com/facebookresearch/fastText)'s
inference (fastText is © Facebook, Inc., MIT License). It loads the same `.bin` and quantized
`.ftz` model files and gives **bit-identical labels and probabilities** to fastText's C++
predict path, as built in fasttext-predict 0.9.2.4 ([parity](../compatibility.md#fasttext)).

!!! note "Verified on Linux aarch64 so far"
    pyroid reproduces the C++ code's floating-point operations exactly, including which
    multiply-adds are fused, and these differ by architecture. Parity on x86-64, macOS and Windows
    is unverified until the CI runs there. Different builds of fastText itself differ slightly
    too: fasttext-numpy2-wheel's probabilities were up to 1.79e-7 away from fasttext-predict's on
    the same inputs.

Two reasons to switch besides speed:

- **It installs everywhere.** Wheels for CPython 3.11–3.15 and free-threaded 3.14t/3.15t on
  Linux (glibc and musl), macOS and Windows, with no compiler. The upstream `fasttext` 0.9.3 is
  an sdist that needs a C++ compiler and breaks on numpy 2, and the wheel-shipping forks stop
  short of some of those interpreters.
- **One model, many threads.** A loaded model is immutable and shared by all threads, and
  batches run without the GIL, so one process with eight threads holds one copy of the model
  instead of eight.

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

The inference methods of fastText's `_FastText` are there with the same results:
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

As in fastText, a line must not contain `"\n"` (`ValueError`); replace newlines first.

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

Deliberate, or harmless for real models (from the module's tests and documentation):

1. **Load errors.** A missing or unreadable file raises `OSError` (`FileNotFoundError`,
   `IsADirectoryError`, ...) where fastText raises `ValueError`. A truncated or corrupt file
   raises [`ModelError`](../reference/fasttext.md#pyroid.fasttext.ModelError) (a `ValueError`)
   where fastText may crash, hang or read garbage.
2. **Limits fastText doesn't have**, which only reject crafted files: `maxn` must be 0–32,
   `wordNgrams` at most 32, and dictionary probe runs at most 1024 (lid.176.bin's longest is 99).
3. **Inputs.** `get_input_vector` out of range raises `IndexError` (undefined behaviour in
   fastText). Tuples and Arrow columns are accepted. A non-`str` item raises
   `pyroid.InputError` (a `TypeError`) where fastText raises `AttributeError`.
4. **Error details.** A lone surrogate gives a `TypeError` with pyroid's own message (same type;
   a `"\n"` still takes precedence, as in fastText). A `k` above 2^63 raises `OverflowError`
   (fastText: `TypeError`); outside the int32 range it is `TypeError`, as in fastText.
5. **Tie order** follows libstdc++, as fastText's Linux builds do. Builds against libc++ or MSVC's
   standard library may order exact ties differently.
6. `get_subwords` returns int64 ids even when empty (fastText gives float64 when every n-gram is
   pruned).

pyroid never delegates to fastText, so it never emits `FallbackWarning`.

## Speed and memory

On the first 10,000 FineWeb documents at `k=1`, pyroid on one thread was 2.1× (`.bin`: 141
against 290 µs per document) and 4.3× (`.ftz`: 105 against 451 µs) faster than fasttext-predict
0.9.2.4. With eight threads in one process it took 20.2 µs (`.bin`) and 16.0 µs (`.ftz`),
holding one model copy instead of eight. *Docker/OrbStack VM on Apple Silicon (12 vCPU),
indicative*: see [Benchmarks](../benchmarks.md#per-stage); `k=5`, load times and memory are in
the [module's earlier run](../benchmarks.md#pyroidfasttext).

API: [`pyroid.fasttext` reference](../reference/fasttext.md).
