# `pyroid.translit`: permissive transliteration

`pyroid.translit` turns Unicode text into ASCII: the `unidecode()` call you know, **without the
GPL**, batched and fast. [Unidecode](https://pypi.org/project/Unidecode/) is GPL-licensed, which
many companies can't ship. pyroid's data comes from [anyascii](https://github.com/anyascii/anyascii)
0.3.3 (© 2020–2025 Hunter WB, ISC License) via the `any_ascii` Rust crate. Nothing in pyroid is
derived from Unidecode's tables.

```python
import pyroid.translit as T

T.anyascii("Björk Guðmundsdóttir")  # 'Bjork Gudmundsdottir'
T.unidecode("北京")  # 'BeiJing'   (Unidecode: 'Bei Jing ')
T.is_ascii(["abc", "ÿ"])  # [True, False]
```

## Functions

| Function | What it does |
|---|---|
| `anyascii(string)` | **byte-identical to the `anyascii` package 0.3.3** on every code point ([parity](../compatibility.md#translit)) |
| `unidecode(string, errors="ignore", replace_str="?")` | **Unidecode 1.4.0's signature and `errors=` semantics**, on top of the same anyascii data, so the output is *not* Unidecode's ([below](#how-different-is-it-from-unidecode)); `unidecode_expect_ascii` and `unidecode_expect_nonascii` are aliases, as in Unidecode |
| `is_ascii(string)` | a mask: `str.isascii()` per document |
| `UnidecodeError` | raised by `errors="strict"`; has Unidecode's `index` attribute and messages |
| `ANYASCII_VERSION` | `"0.3.3"`, the anyascii data version |

All three functions take a `str`, a `list` or `tuple` of `str`, or an Arrow string column, return
the same kind (Arrow nulls stay null), return pure-ASCII documents as the original objects, and
run batches on pyroid's thread pool without the GIL (`threads=`, see
[Threads](../concepts/threads.md)). For `str`, `list` and `tuple` input, `is_ascii` reads
CPython's own ASCII flag without scanning the text, so `threads` only matters for Arrow input.

## Replacing Unidecode

```diff
-from unidecode import unidecode
+from pyroid.translit import unidecode

 unidecode("Björk Guðmundsdóttir")              # 'Bjork Gudmundsdottir', as Unidecode
 unidecode("Ærøskøbing")                        # 'Aeroskobing' (Unidecode: 'AEroskobing')
 unidecode(text, errors="replace", replace_str="?")
```

`errors=` works as in Unidecode, for the characters that have no transliteration:

- `"ignore"` (default): leave them out.
- `"replace"`: put `replace_str` in their place.
- `"strict"`: raise `UnidecodeError`, whose `index` is the character's position.
- `"preserve"`: keep the character (the result is then not pure ASCII).

As in Unidecode, `errors` and `replace_str` are only checked when such a character is met, and each
lone surrogate gives a `RuntimeWarning` ("Surrogate character ... will be ignored. ...") and then
counts as a character without a mapping. For a batch, the first failing document raises, with a
note naming it.

**The transliterations themselves differ.** If you need Unidecode's exact output, keep Unidecode.
If you need reasonable ASCII under a permissive licence, the measurements below show how far
apart they are.

### Which characters have no transliteration

`errors=` applies to characters anyascii has no mapping for: unassigned and private-use code
points, lone surrogates, and the characters of scripts or blocks anyascii does not cover (for
example most Tangut, Khitan and some CJK extension ideographs). anyascii's table lists the 123,799
non-ASCII code points it maps, 1,192 of them to the empty string on purpose (combining marks,
zero-width joiners, variation selectors, emoji skin-tone modifiers, Arabic tatweel, silent vowel
carriers): those are removed in every mode and never raise.

## How different is it from Unidecode?

Measured against Unidecode 1.4.0 as a black box, both with `errors="ignore"`, on CPython 3.15
(Unicode 17.0), on every code point and on FineWeb and FineWeb-2 documents.

**Per code point:** of the 42,985 non-ASCII code points both libraries map, 6,705 (15.6%) give
identical output, and 58.8% after stripping spaces (Unidecode often adds a trailing space, e.g.
after CJK ideographs). Unidecode has no mapping for 80,814 of the 123,799 non-ASCII code points
anyascii maps (most of the CJK extensions, emoji and the supplementary planes). No assigned
character has a Unidecode mapping but no anyascii one.

**Per document:**

| Corpus | Documents with identical output | Non-ASCII characters mapped identically |
|---|---:|---:|
| FineWeb, 100k English web documents | 79.7% (68% of those not pure ASCII) | 86.5% |
| FineWeb-2 French / Spanish (250 docs each) | 90.4% / 90.4% | 99.7% / 99.3% |
| FineWeb-2 German | 75.2% | 97.9% |
| FineWeb-2 Russian / Arabic / Hindi / Japanese / Chinese | 0% each | 96.0% / 88.7% / 77.0% / 60.2% / 5.6% |

Whole non-Latin documents almost never match, but most of their characters do; Chinese differs
mostly by Unidecode's trailing spaces.

The most frequent differences on that text:

| Character | Unidecode 1.4.0 | pyroid |
|---|---|---|
| U+2014 EM DASH `—` | `--` | `-` |
| U+093E DEVANAGARI VOWEL SIGN AA `ा` | `aa` | `a` |
| U+0940 DEVANAGARI VOWEL SIGN II `ी` | `ii` | `i` |
| U+0629 `ة` (Arabic tā’ marbūṭa) | `@` | `h` |
| U+044F CYRILLIC SMALL LETTER YA `я` | `ia` | `ya` |
| U+3002 IDEOGRAPHIC FULL STOP `。` | `. ` | `.` |
| U+7684 CJK UNIFIED IDEOGRAPH `的` | `De ` | `De` |
| U+00A3 POUND SIGN `£` | `PS` | `GBP` |
| U+00AE REGISTERED SIGN `®` | `(r)` | `(R)` |

## Differences from Unidecode

From the module's documentation (`pyroid/translit.pyi`):

- **Transliterations differ** (anyascii's are not Unidecode's). For example, CJK ideographs are
  joined without spaces (`"北京"` → `"BeiJing"`, where Unidecode gives `"Bei Jing "`), emoji
  become `:shortcodes:` (Unidecode drops them), and many scripts anyascii covers have no
  Unidecode mapping.
- **Which characters are missing differs**, so `errors="strict"` raises, and `errors="replace"`
  substitutes, on different characters. Unidecode has no mapping for 80,814 of the 123,799
  non-ASCII code points anyascii maps (most of the CJK extensions, emoji and the supplementary
  planes). No assigned character has a Unidecode mapping but no anyascii one.
- Lists, tuples and Arrow columns are accepted (Unidecode raises `AttributeError`); other non-str
  inputs raise `pyroid.InputError` (a `TypeError`) instead of `AttributeError`.
- For a batch, the first failing document raises, with a note naming it.
- `UnidecodeError` is pyroid's own class (a `ValueError` and a `pyroid.PyroidError`), not an
  instance of Unidecode's.
- The surrogate `RuntimeWarning` is attributed to the caller's line rather than to a line inside
  the library.

pyroid never delegates to Unidecode or anyascii, so it never emits `FallbackWarning`.

## Speed

On the first 10,000 FineWeb documents, `anyascii` took 1.62 µs per document on one thread
(free-threaded CPython 3.15t) against 128 µs for the `anyascii` 0.3.3 package (CPython 3.13),
79× faster, and 1.00 µs with 8 threads; on Arrow input, 1.15 µs on one thread and 0.41 µs on 8.
*Docker/OrbStack VM on Apple Silicon (12 vCPU), indicative*: see
[Benchmarks](../benchmarks.md#per-stage). The module's earlier run (CPython 3.12, fresh `str`
objects) measured 12–37× against the `anyascii` package and 20–86× against Unidecode across
FineWeb and FineWeb-2 languages, and Arrow input scaled up to 6.1× at 8 threads on 3.15t
(*indicative, preliminary*, [details](../benchmarks.md#pyroidtranslit)).

API: [`pyroid.translit` reference](../reference/translit.md).
