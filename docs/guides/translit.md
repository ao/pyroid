# `pyroid.translit`: permissive transliteration *(preview)*

!!! warning "Preview"
    `pyroid.translit` is still being finished and is not in the main branch yet. This page is
    written from the module's draft documentation and its working notes; names and details may
    change before release. Its API reference will be generated here once it is merged.

`pyroid.translit` turns Unicode text into ASCII: the `unidecode()` call you know, **without the
GPL**, batched and fast. [Unidecode](https://pypi.org/project/Unidecode/) is GPL-licensed, which
many companies can't ship. pyroid's data comes from [anyascii](https://github.com/anyascii/anyascii)
0.3.3 (© Hunter WB, ISC License) via the `any_ascii` Rust crate. Nothing in pyroid is derived from
Unidecode's tables.

```python
import pyroid.translit as T

T.anyascii("Björk Guðmundsdóttir")  # 'Bjork Gudmundsdottir'
T.unidecode("北京")  # 'BeiJing'   (Unidecode: 'Bei Jing ')
T.is_ascii(["abc", "ÿ"])  # [True, False]
```

## Functions

| Function | What it does |
|---|---|
| `anyascii(text)` | **byte-identical to the `anyascii` package 0.3.3** on every code point |
| `unidecode(text, errors="ignore", replace_str="?")` | **Unidecode 1.4.0's signature and `errors=` semantics**, on top of the same anyascii data, so the output is *not* Unidecode's (below) |
| `is_ascii(text)` | a mask: `str.isascii()` per document |

All three take a `str`, a `list` or `tuple` of `str`, or an Arrow string column, return the same
kind, keep pure-ASCII documents as the original objects, and run batches on pyroid's thread pool
without the GIL (`threads=`, see [Threads](../concepts/threads.md)).

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
lone surrogate gives a `RuntimeWarning` and then counts as a character without a mapping.

**The transliterations themselves differ.** If you need Unidecode's exact output, keep Unidecode.
If you need "reasonable ASCII under a permissive licence", the numbers below show how far apart
they are.

## How different is it from Unidecode?

Measured on every code point and on FineWeb and FineWeb-2 documents, both with
`errors="ignore"` (from the translit session's agreement run against Unidecode 1.4.0 as a black
box; CPython 3.15, Docker VM).

| Corpus | Documents with identical output |
|---|---:|
| FineWeb, 100k English web documents | 79.67% |
| FineWeb-2 French / Spanish (250 docs each) | 90.40% / 90.40% |
| FineWeb-2 German | 75.20% |
| FineWeb-2 Arabic, Chinese, Hindi, Japanese, Russian | 0.00% |

Whole non-Latin documents almost never match, but most characters do in several scripts: per
document, the mean share of non-ASCII characters transliterated identically is 95.96% for Russian,
88.74% for Arabic, 76.95% for Hindi, 60.16% for Japanese and 5.55% for Chinese (anyascii joins CJK
syllables without Unidecode's trailing spaces).

Some of the most frequent differences on these corpora (the emoji is from the Spanish slice):

| Character | Unidecode 1.4.0 | pyroid |
|---|---|---|
| U+2014 EM DASH `—` | `--` | `-` |
| U+093E DEVANAGARI VOWEL SIGN AA `ा` | `aa` | `a` |
| U+0629 `ة` (Arabic tā’ marbūṭa) | `@` | `h` |
| U+044F CYRILLIC SMALL LETTER YA `я` | `ia` | `ya` |
| U+3002 IDEOGRAPHIC FULL STOP `。` | `. ` | `.` |
| U+7684 CJK UNIFIED IDEOGRAPH `的` | `De ` | `De` |
| U+00A3 POUND SIGN `£` | `PS` | `GBP` |
| U+00AE REGISTERED SIGN `®` | `(r)` | `(R)` |
| U+2728 SPARKLES `✨` | (removed) | `:sparkles:` |

Coverage also differs: anyascii maps 123,927 code points and Unidecode 43,120 (ASCII included),
so `errors="strict"` raises on different characters. Unidecode has no mapping for most of the CJK
extensions and the supplementary planes that anyascii covers.

## Differences from Unidecode

From the module's draft documentation:

- **Transliterations differ** (anyascii's are not Unidecode's). For example, CJK ideographs are
  joined without spaces (`"北京"` → `"BeiJing"`, where Unidecode gives `"Bei Jing "`), emoji
  become `:shortcodes:` (Unidecode drops them), and many scripts anyascii covers have no
  Unidecode mapping.
- **Which characters are missing differs**, so `errors="strict"` raises, and `errors="replace"`
  substitutes, on different characters.
- Lists, tuples and Arrow columns are accepted (Unidecode raises `AttributeError`); other non-str
  inputs raise `pyroid.InputError` (a `TypeError`) instead of `AttributeError`.
- For a batch, the first failing document raises, with a note naming it.
- `UnidecodeError` is pyroid's own class (a `ValueError` and a `pyroid.PyroidError`), not an
  instance of Unidecode's.
- The surrogate `RuntimeWarning` is attributed to the caller's line rather than to a line inside
  the library.

## Speed

On the first 10k FineWeb documents, single-threaded on CPython 3.12, `pyroid.translit.anyascii`
took 2.19 µs per document against 80.77 µs for the `anyascii` package, and `unidecode` 2.23 µs
against 192.00 µs for Unidecode. *Docker/OrbStack VM on Apple Silicon, indicative, preliminary*,
from the translit session's working notes: see [Benchmarks](../benchmarks.md#pyroidtranslit-preview).
