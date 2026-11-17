# `pyroid.langdetect`: drop-in langdetect

`pyroid.langdetect` is a Rust port of [langdetect](https://github.com/Mimino666/langdetect)
1.0.9 by Michal Danilak (Apache-2.0), itself a port of Cybozu Labs' language-detection. It
returns the same language codes and the same probabilities, to the last bit, for the same
`DetectorFactory.seed` ([parity](../compatibility.md#langdetect)), 40 to 65 times faster per
call, and detects batches on all cores. The other fast detectors (fastText's lid.176, lingua,
fast-langdetect) are different models and give different answers; this one keeps langdetect's.

## Change one import

```diff
-from langdetect import DetectorFactory, detect, detect_langs
+from pyroid.langdetect import DetectorFactory, detect, detect_langs

 DetectorFactory.seed = 0
 detect("War doesn't show who's right, just who's left.")   # 'en'
 detect_langs("Otec matka syn.")                             # [pl:0.8571394483837576, fi:0.14285787958897214]
```

Or keep the import and patch langdetect itself (see [below](#patch)):

```python
import pyroid

pyroid.patch("langdetect")
from langdetect import detect  # langdetect's own function, running pyroid's code
```

Everything langdetect 1.0.9 has is there: `detect`, `detect_langs`, `DetectorFactory` (`seed`,
`load_profile`, `load_json_profile`, `add_profile`, `create`, `get_lang_list`, `set_seed`,
`clear`), `Detector` (`append`, `detect`, `get_probabilities`, `set_alpha`,
`set_max_text_length`, `set_prior_map`, `set_verbose`), `Language`, `LangDetectException`
with `ErrorCode`, `LangProfile`, `NGram`, `init_factory` and `PROFILES_DIRECTORY`, under the
same submodule names (`pyroid.langdetect.detector_factory`, `.detector`, `.language`,
`.lang_detect_exception`, `.utils.lang_profile`, `.utils.ngram`).

With langdetect installed, `Language`, `LangDetectException` and `ErrorCode` are langdetect's
own classes. Code that catches `langdetect.LangDetectException` or checks
`isinstance(x, langdetect.language.Language)` works unchanged with pyroid's results.

## Seeds and randomness

langdetect is random: each detection runs 7 trials that sample n-grams of the text. With
`DetectorFactory.seed` set, the samples are the same every time. pyroid draws the same samples:
it implements CPython's Mersenne Twister and `random.Random`'s `gauss`, `choice` and seeding
(ints, and through `random.Random` itself, str, bytes, bytearray and float seeds), and calls
them in langdetect's order, with langdetect's floating-point operations in langdetect's order.
That includes `sum()`, which CPython 3.12 changed to a compensated sum: pyroid adds as the
running interpreter does, so results equal langdetect's on 3.11 and on 3.12 and later.

Without a seed (`seed = None`, the default), langdetect gives each detector its own
`random.Random()` and reseeds it from the operating system's random source for every
detection; it never touches the program's global `random` state. pyroid does the same: a fresh
generator per detection, seeded from `os.urandom`, and the global `random` state stays
unchanged. So the results without a seed have the same distribution as langdetect's, but, as
with langdetect itself, two runs (of either library) do not give equal probabilities.

## Many texts: `detect_batch`

```python
import pyroid.langdetect as ld

ld.DetectorFactory.seed = 0
ld.detect_batch(["Das ist gut.", "C'est bon.", ""], errors="none")  # ['de', 'fr', None]
ld.detect_langs_batch(titles)  # one list of Language per text
ld.detect_batch(table["text"])  # an Arrow column -> pyroid.Column
```

`detect_batch(texts)` is `[detect(t) for t in texts]` and `detect_langs_batch(texts)` is
`[detect_langs(t) for t in texts]`: every text is detected as a separate call would detect it,
with the default factory and `DetectorFactory.seed`, so a batch gives exactly the loop's
results, whatever the number of threads. Texts are detected on pyroid's shared thread pool
(`threads=` caps it; `threads=1` uses the calling thread), in chunks of about 50 ms with Ctrl-C
checked between them.

- `texts`: a list or tuple of str (a tuple gives a tuple), or an Arrow string array
  (pyarrow, polars, ...; [inputs](../concepts/inputs.md)). `detect_batch` on Arrow input
  returns a [`pyroid.Column`](../reference/pyroid.md#pyroid.Column) of strings, with nulls kept.
- `errors="raise"` (the default): the first text that langdetect cannot detect (empty, only
  digits or punctuation, only a URL) raises `LangDetectException`, with a note naming its
  index. `errors="none"` gives `None` for it instead.

## Custom profiles and the factory

```python
from pyroid.langdetect import DetectorFactory

factory = DetectorFactory()
factory.load_profile("/path/to/profiles")  # langdetect's JSON profile files
factory.load_json_profile([json_a, json_b])
detector = factory.create(alpha=0.3)
detector.set_prior_map({"en": 0.7, "fr": 0.3})
detector.append(text)
detector.get_probabilities()
```

Profiles are read as langdetect reads them (`os.listdir` order, `json`, `LangProfile`), into a
table in Rust; every error has langdetect's code and message, and langdetect's quirks are kept
(for example, two `load_profile` calls on one factory both number their languages from 0, so
the second set's rows are shorter than the language list and detection raises `IndexError`, as
in langdetect). `word_lang_prob_map` is a read-only view of that table.

The default profiles (`detect`, `detect_langs`, `init_factory`) are a copy of langdetect
1.0.9's 55 profiles in `pyroid.langdetect.PROFILES_DIRECTORY`; loading them takes 0.1 s
(langdetect: 0.16 s). langdetect loads profiles in the order `os.listdir` returns, which
depends on the file system, and that order decides `get_lang_list()`, ties between equal
probabilities and the order of the floating-point sums, so it can change the last bits of the
probabilities from one machine to another, in langdetect too. pyroid uses the order of
langdetect's own profile directory when langdetect is installed, and of its copy otherwise.

## What runs where

`append` (the URL and mail-address removal, the Vietnamese normalization, the cut at
`max_text_length`), the n-gram extraction and the trials run in Rust, on code points, so lone
surrogates are handled as langdetect handles them. langdetect runs its two regular
expressions over the whole text before cutting it at 10,000 characters; pyroid processes only
the part that can reach the cut, so long documents cost little more than 10,000 characters.

Two kinds of detection run langdetect's own algorithm in Python instead, with the same
results and langdetect's speed: `set_verbose()` (it prints every step), and detectors whose
attributes have types langdetect never sets (a `Decimal` alpha, a seed of another type, a
`prior_map` that is not a list of floats). No call is ever handed to langdetect itself.

## `pyroid.patch("langdetect")` {#patch}

With langdetect installed, `pyroid.patch("langdetect")` (or `pyroid run -p langdetect
script.py`) gives langdetect's own `Detector` pyroid's `append` and `_detect_block`, and makes
`langdetect.detect` and `langdetect.detect_langs` one native call when the default factory and
classes are langdetect's own. Everything stays langdetect's: its classes, its exceptions, its
`Language` objects, its factory and `DetectorFactory.seed`. Factories and detectors made before
the patch are sped up too, and `pyroid.unpatch("langdetect")` restores the original methods;
each method is an exact replacement on langdetect's own object state, so calls running in other
threads while patching or unpatching still give langdetect's results. The first detection of
each langdetect factory builds pyroid's table from its `word_lang_prob_map` (about 0.1 s for the
default profiles).

Without langdetect, `pyroid.patch("langdetect")` makes `import langdetect` (and its
submodules) give pyroid's classes.

## Differences from langdetect

- `word_lang_prob_map` is a read-only view: changing the lists it returns does not change
  detections (add profiles with `add_profile` instead). Under the patch, langdetect's own dict
  is used and changes that do not change its size or the language list are not seen after the
  first detection.
- Overriding `Detector`'s private methods (`_update_lang_prob`, `_normalize_prob`, ...) in a
  subclass does not change `detect()`, which does not call them.
- `Detector.random` exists, but pyroid's detections draw from their own generator, seeded the
  same way, so its state does not advance.
- Profiles are parsed with `json`, never `simplejson` (the results are the same for valid
  profiles).
- Ctrl-C while `load_profile` reads the profiles raises `KeyboardInterrupt`; langdetect turns
  it into a `LangDetectException`. A failed first `init_factory()` is retried by the next call;
  langdetect keeps the half-loaded factory.
- A frequency in a custom profile that is not an int or a float is converted with `float()`
  after langdetect's arithmetic.

## Parity

3.7 million comparisons of `detect` and `detect_langs` (every probability compared as
`float.hex`) between langdetect 1.0.9 and pyroid on FineWeb and the FineWeb-2 test splits of all
55 profile languages: whole documents, the first 50 characters, tweets, titles, 3-word queries
and two-language mixes, with 20 seeds, on CPython 3.11, 3.12, 3.15 and free-threaded 3.14t.
None differed, and the batch functions gave the per-call results for every text. Details in
[Compatibility](../compatibility.md#langdetect).

## Speed

µs per text, median of 5 rounds, every text's `detect_langs` first checked against
langdetect's (EC2 c8g.16xlarge, shared with other jobs, so indicative; CPython 3.12, release
wheel; texts from FineWeb and FineWeb-2 in all profile languages, round robin;
`bench/langdetect_bench.py`):

| Run | 50 characters | 1,500 characters | 12,000 characters |
|---|---:|---:|---:|
| langdetect, one thread | 1,692 | 3,278 | 18,894 |
| langdetect, 8 processes | 246 (7×) | 425 (8×) | 2,440 (8×) |
| pyroid, one call per text | 39 (43×) | 59 (55×) | 293 (64×) |
| pyroid `detect_batch(texts, threads=1)` | 37 (46×) | 55 (60×) | 279 (68×) |
| pyroid `detect_batch(texts, threads=8)` | 5.6 (300×) | 7.8 (420×) | 38 (496×) |

Short texts cost the most per character: langdetect's trials converge slowly on them (about 750
sampling steps per short English text, against 300 for 1,500 characters). langdetect's own `detect` under `pyroid.patch("langdetect")` takes
57 µs on the 1,500-character texts (57×). A fresh interpreter reaches its first detection in
120 ms with pyroid (import and profile loading) and 231 ms with langdetect.
