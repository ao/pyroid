# `pyroid.langdetect`: drop-in langdetect

`pyroid.langdetect` is a Rust port of [langdetect](https://github.com/Mimino666/langdetect)
1.0.9 by Michal Danilak (Apache-2.0), itself a port of Cybozu Labs' language-detection. It
returns the same language codes and the same probabilities, to the last bit, for the same
`DetectorFactory.seed` ([parity](../compatibility.md#langdetect)), 35 to 50 times faster per
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
detector = factory.create(alpha=0.3)
detector.set_prior_map({"en": 0.7, "fr": 0.3})
detector.append(text)
detector.get_probabilities()

from_strings = DetectorFactory()  # JSON strings go in a factory of their own
from_strings.load_json_profile([json_a, json_b])
```

Profiles are read as langdetect reads them (`os.listdir` order, `json`, `LangProfile`), into a
table in Rust; every error has langdetect's code and message, and langdetect's quirks are kept.
For example, each `load_profile` or `load_json_profile` call numbers its languages from 0, so
after a second load into the same factory the second set's rows are shorter than the language
list and detection raises `IndexError`, as in langdetect: use one loader per factory.
`word_lang_prob_map` is a read-only view of that table; assigning a new map
(`factory.word_lang_prob_map = {...}`) works as in langdetect.

Profiles whose counts are not ints or floats, such as numpy numbers, give probabilities that are
not floats in langdetect (`numpy.float64`), and its sums then add them one by one rather than
with the compensated `sum()` of CPython 3.12. Such a factory keeps langdetect's dict instead of the
Rust table, and its detections run langdetect's Python algorithm, so they give langdetect's
`numpy.float64` results at langdetect's speed.

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
expressions over the whole text before cutting it at 10,000 characters; pyroid reads and
processes only the part that can reach the cut, so a long document costs little more than
10,000 characters, whatever characters it holds.

Some detections run langdetect's own code instead, in pyroid's line-for-line Python port, with
langdetect's results; no call is ever handed to langdetect itself. A pattern changed
(`Detector.URL_RE` or `MAIL_RE`, on the class or on a detector) moves only `append`'s
substitutions to Python (47 µs for a tweet instead of 32 µs). The others run langdetect's whole
algorithm in Python, at about langdetect's speed:

- `set_verbose()`, which prints every step.
- Attributes of types langdetect never sets: a `Decimal` alpha, a seed of another type, a
  `prior_map` that is not a list of floats.
- langdetect's classes used as hooks, which the Rust code would ignore:
    - a `Detector` method (`cleaning_text`, `_extract_ngrams`, `_update_lang_prob`, ...)
      overridden in a subclass, replaced on the class or set on a detector. A subclass that only
      overrides `cleaning_text` runs it, then the rest in Rust;
    - a `detector.random` that is not a `random.Random`, or a `word_lang_prob_map` of another
      object;
    - `NGram`'s tables or methods changed: `N_GRAM`, `LATIN1_EXCLUDED`, `CJK_MAP` (also edited in
      place), the Vietnamese tables, `normalize`, `add_char`.
- Profiles whose probabilities are not floats (above).

With `DetectorFactory.create` overridden, or `Detector` or `DetectorFactory` replaced in their
modules, `detect`, `detect_langs` and the batch functions take langdetect's steps (`create()`,
`append()`, `detect()`), one text at a time, and each step runs as above.

Checking for these hooks takes well under a microsecond per call, in Rust: about 0.4 µs for
`detect`, 0.8 µs for a detector made with `create()` (EC2 c8g.16xlarge devbox, shared, CPython
3.12; the timings above too). A detection that uses none of them always takes the Rust path.

## `pyroid.patch("langdetect")` {#patch}

With langdetect installed, `pyroid.patch("langdetect")` (or `pyroid run -p langdetect
script.py`) gives four of langdetect's own functions pyroid's code: `detect` and
`detect_langs` take one native call, and `Detector.append` and `Detector._detect_block` run in
Rust on langdetect's own object state. The functions stay the same objects (their code is
swapped), and everything else stays langdetect's: its classes, its exceptions, its `Language`
objects, its factory and `DetectorFactory.seed`. So:

- names imported before the patch (`from langdetect import detect`) and factories and detectors
  made before it are sped up too;
- the patched functions pickle as langdetect's: a process without the patch (a spawn or
  forkserver worker, a joblib worker) loads langdetect's own function and runs langdetect's own
  code, with the same results;
- customised langdetect classes (the hooks listed [above](#what-runs-where)) run langdetect's
  own code;
- `pyroid.unpatch("langdetect")` puts the original code back. Each replacement gives
  langdetect's results on langdetect's own object state, so calls running in other threads
  while patching or unpatching still give langdetect's results.

The first detection of each langdetect factory builds pyroid's table from its
`word_lang_prob_map` (about 0.1 s for the default profiles).

Without langdetect, `pyroid.patch("langdetect")` makes `import langdetect` (and its
submodules) give pyroid's classes. `langdetect.detector_factory` reads and sets `_factory`,
`PROFILES_DIRECTORY` and the other names langdetect's code reads on
`pyroid.langdetect.detector_factory`, so `detector_factory._factory` exists after
`init_factory()` and a factory assigned to it is the one `langdetect.detect` uses. A
`DetectorFactory` or `Detector` pickled by langdetect loads as pyroid's.

## langdetect releases

pyroid reproduces langdetect 1.0.9. Releases 1.0.7 and 1.0.8 have the same detection code and
profiles, and detect the same. With another release installed, `import pyroid.langdetect` (and
so `pyroid.patch("langdetect")`) warns once per process with a `RuntimeWarning`: detections
follow 1.0.9, and `Language`, `LangDetectException` and `ErrorCode` are the installed
release's classes. To silence it:

```python
import warnings

warnings.filterwarnings("ignore", message="pyroid.langdetect reproduces", category=RuntimeWarning)
```

## Differences from langdetect

- `word_lang_prob_map` is a read-only view: changing the lists it returns does not change
  detections (add profiles with `add_profile`, or assign a new map). Under the patch,
  langdetect's own dict is used and changes that do not change its size or the language list
  are not seen after the first detection. Under the patch, entries of langdetect's
  `NGram.CJK_MAP` changed in place are not seen either (a new dict assigned to it is); pyroid's
  own `NGram.CJK_MAP` is a `dict` subclass that records changes.
- `Detector.random` exists, but pyroid's detections draw from their own generator, seeded the
  same way, so its state does not advance (one of another class than `random.Random` is used).
- Profiles are parsed with `json`, never `simplejson` (the results are the same for valid
  profiles).
- Ctrl-C while `load_profile` reads the profiles raises `KeyboardInterrupt`; langdetect turns
  it into a `LangDetectException`. A failed first `init_factory()` is retried by the next call;
  langdetect keeps the half-loaded factory.
- pyroid's `DetectorFactory` and `Detector` pickle in pyroid's own form (the table as bytes),
  which only pyroid loads.

## Parity

3.7 million comparisons of `detect` and `detect_langs` (every probability compared as
`float.hex`) between langdetect 1.0.9 and pyroid on FineWeb and the FineWeb-2 test splits of all
55 profile languages: whole documents, the first 50 characters, tweets, titles, 3-word queries
and two-language mixes, with 20 seeds, on CPython 3.11, 3.12, 3.15 and free-threaded 3.14t.
None differed, and the batch functions gave the per-call results for every text. Details in
[Compatibility](../compatibility.md#langdetect).

## Speed

µs per text, median of 5 rounds, every text's `detect_langs` first checked against
langdetect's (a dedicated AWS EC2 c7i.4xlarge, Intel Xeon Platinum 8488C, 16 vCPU; CPython
3.12.16, release wheel; texts from FineWeb and FineWeb-2 in all profile languages, round robin;
`bench/langdetect_bench.py`; the [benchmarks page](../benchmarks.md#pyroidlangdetect) has the
details):

| Run | 50 characters | 1,500 characters | 12,000 characters |
|---|---:|---:|---:|
| langdetect, one thread | 1,366 | 2,687 | 15,191 |
| langdetect, 8 processes | 200 (7×) | 348 (8×) | 1,994 (8×) |
| pyroid, one call per text | 39 (35×) | 58 (46×) | 309 (49×) |
| pyroid `detect_batch(texts, threads=1)` | 37 (37×) | 53 (51×) | 289 (52×) |
| pyroid `detect_batch(texts, threads=8)` | 8.7 (158×) | 12 (220×) | 62 (246×) |

Short texts cost the most per character: langdetect's trials converge slowly on them (about 750
sampling steps per short English text, against 300 for 1,500 characters). langdetect's own `detect` under `pyroid.patch("langdetect")` takes
55 µs on the 1,500-character texts (49×). A fresh interpreter reaches its first detection in
112 ms with pyroid (import and profile loading) and 237 ms with langdetect.

The c7i's 16 vCPUs are 8 cores with two hardware threads each, and `threads=8` runs on 8 of the
pool's 16 threads, so some of them share a core. Pinned to one thread per core
(`taskset -c 0-7`), the 12,000-character batch took 41 to 46 µs per text instead of 59 to 61.
