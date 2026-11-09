# Languages

pyroid's filters and dedup reproduce [datatrove](https://github.com/huggingface/datatrove) 0.10.1
beyond English. datatrove picks a word tokenizer for each of its 4,343 language codes (its
`tokenizer_assignment.csv`), and every word count, ratio, n-gram and shingle depends on that
tokenizer's words. pyroid ports the rule-based tokenizers, jieba for Chinese and SudachiPy for
Japanese to Rust. That makes 4,013 of the 4,343 codes native, 95.6% of FineWeb-2's bytes, with
datatrove's decisions ([parity](#parity)), in Rust on every thread. Japanese needs a SudachiDict
dictionary installed ([Japanese](#japanese)); without one, 4,009 codes (88.4%) are native. The
other codes use statistical or dictionary segmenters that pyroid does not port. For them,
`python_tokenizer=True` calls datatrove's own tokenizer and runs the rest in Rust.

The largest languages are native: Russian, German, Spanish, French and Arabic together are 48% of
FineWeb-2's bytes, Chinese another 11.6% and Japanese 7.2%.

## The language table

datatrove 0.10.1's tokenizer for each code, and the share of FineWeb-2's training data (1,913
subsets, 22.9 TB) in those codes:

| datatrove tokenizer | codes | FineWeb-2 bytes | pyroid |
|---|--:|--:|---|
| `SpaCyTokenizer(code)`, a rule-based `spacy.blank(code)`: the 57 spaCy languages `af` … `yo`, including `xx`, spaCy's multi-language tokenizer (1,995 codes such as `uzn_Latn`) | 3,742 | 74.8% | native |
| `IndicNLPTokenizer`: indic-nlp-library's 16 languages (`hi bn ur ta te mr ne ml kn gu si pa or as sa kK`) | 129 | 1.9% | native (words and sentences) |
| `WhitespaceTokenizer`: spaCy `xx` words and datatrove's regex sentences (such as `kat_Geor`) | 128 | 0.1% | native |
| `SpaCyTokenizer("zh")`: spaCy's Chinese tokenizer with jieba (such as `cmn_Hani`, `yue_Hani`) | 10 | 11.6% | native ([`pyroid.jieba`](jieba.md)) |
| `SpaCyTokenizer("ja")`: spaCy's Japanese pipeline with datatrove's SudachiPy tokenizer (`ja`, `ja_Jpan`, `jpn`, `jpn_Jpan`) | 4 | 7.2% | native with an installed SudachiDict ([Japanese](#japanese)), else `python_tokenizer=True` |
| `SpaCyTokenizer("vi")`: spaCy's Vietnamese tokenizer with pyvi (such as `vie_Latn`) | 16 | 1.5% | `python_tokenizer=True` |
| `StanzaTokenizer` (255 codes), `ThaiTokenizer` (20), `BurmeseTokenizer` (14), `TibetanTokenizer` (10), `LaoTokenizer` (6), `KhmerTokenizer` (5), `KiwiTokenizer` (4) | 314 | 2.7% | `python_tokenizer=True` |
| none: FineWeb-2's `und_*` subsets (except `und` and `und_Cyrl`, which datatrove gives `xx`) | — | 0.15% | `ValueError`, as datatrove |

- Native codes are tokenized in Rust. For the spaCy languages pyroid ports spaCy 3.8.18's
  prefix, suffix, infix and URL rules, each language's tokenizer exceptions, the special-case
  matcher and the sentencizer, extracted from spaCy's own objects (48,032 rules). For the Indic
  languages it ports indic-nlp-library's `trivial_tokenize` and `sentence_split`, and for the
  whitespace codes datatrove's `WhitespaceTokenizer`. Chinese is spaCy's `ChineseTokenizer` over
  [`pyroid.jieba`](jieba.md), the port of jieba 0.42.1, with jieba's default dictionary.
  Japanese is datatrove's `JapaneseTokenizer` (SudachiPy in split mode A) and the sentencizer,
  over a port of SudachiPy 0.7.0's tokenizer that reads an installed SudachiDict dictionary.
  Words and sentences match datatrove's.
- The `python_tokenizer=True` codes need a segmenter pyroid does not port: the statistical models
  of pyvi, Stanza and khmer-nltk, PyThaiNLP, Kiwi, botok and pyidaungsu. Exact ports of these are
  not feasible at a reasonable binary size. Without `python_tokenizer=True` these codes raise
  `NotImplementedError`, and so do the Japanese codes when no dictionary is installed.
- Codes datatrove does not know raise datatrove's `ValueError` when the call starts.

## How to choose

Look the code up in `pyroid.filters.languages()` (the same function is `pyroid.dedup.languages`).
Its keys are what `language=` accepts, in datatrove's spellings (`"deu_Latn"`, `"deu"`, `"de"`,
...). Each value says which tokenizer datatrove uses and whether pyroid's is native (here with a
Japanese dictionary installed; without one, `jpn_Jpan` is `native=False` and the sum is 4009):

```python
import pyroid.filters as F

langs = F.languages()  # {code: Language(tokenizer, arg, native)}, sorted by code
langs["deu_Latn"]  # Language(tokenizer='SpaCyTokenizer', arg='de', native=True)
langs["hin_Deva"]  # Language(tokenizer='IndicNLPTokenizer', arg='hi', native=True)
langs["kat_Geor"]  # Language(tokenizer='WhitespaceTokenizer', arg='', native=True)
langs["cmn_Hani"]  # Language(tokenizer='SpaCyTokenizer', arg='zh', native=True)
langs["jpn_Jpan"]  # Language(tokenizer='SpaCyTokenizer', arg='ja', native=True)
langs["vie_Latn"]  # Language(tokenizer='SpaCyTokenizer', arg='vi', native=False)
sum(lang.native for lang in langs.values())  # 4013 (of 4343)
F.english_languages()[:4]  # ['en', 'en_Latn', 'eng', 'eng_Latn']: SpaCyTokenizer("en")
```

Then:

1. If `native=True`, pass `language=`. Nothing else is needed (no datatrove, no spaCy), and the
   work runs in Rust on every thread.
2. If `native=False`, pass `language=` and `python_tokenizer=True`, with datatrove 0.10.1 and
   the segmenter installed. The `NotImplementedError` you get without them gives the
   `pip install` line. (For the Japanese codes, `pip install sudachidict-core` makes them native
   instead.) Each document is tokenized by datatrove's own tokenizer, in Python, on the
   calling thread, so the call runs at that tokenizer's speed and does not scale with
   `threads=`. On 8 cores datatrove's 8 processes can beat it ([Speed](#speed)). For a large
   corpus in such a language, run several processes, or keep datatrove for those steps.
3. If the code is not in the table, the call raises datatrove's `ValueError`. Pass a code
   datatrove knows, or a tokenizer object (below).

`language=` may also be a word tokenizer object, as datatrove allows: a datatrove `WordTokenizer`
instance, or any object with `word_tokenize(text)` (and `span_tokenize(text)` for C4). It is
called in Python, as on the `python_tokenizer=True` path and with the same `FallbackWarning`, but
needs no `python_tokenizer=True`.

## What takes `language=`

| API | `language=` | `python_tokenizer=` |
|---|---|---|
| [`pyroid.filters`](filters.md): `gopher_quality`, `gopher_repetition`, `c4_quality`, `c4_lines`, `fineweb_quality`, their `*_reasons`, and the configs (`GopherQuality(language=...)`, ...) | any datatrove code, or a tokenizer object | yes |
| `pyroid.filters.apply` | each step's own `language` | yes, for every step |
| [`pyroid.dedup`](dedup.md): `minhash_signatures`, `minhash_dedup`, `MinHashDedup` | any datatrove code, or a tokenizer object | yes |
| `pyroid.dedup.sentence_dedup`, `SentenceDedup` | native codes only (a `str`) | no: other codes raise `NotImplementedError` |
| `pii_replace`, `symbol_lines`, `URLFilter`, `url_filter` | no tokenizer | — |
| `c4_badwords`, `BadWordsFilter` | `languages=`: each document's word-list language (datatrove's `doc.metadata["language"]`) | — |

What `python_tokenizer=True` does, from the modules' documentation (`pyroid/filters.pyi`,
`pyroid/dedup.pyi`):

- Each document is tokenized by datatrove's own tokenizer for `language`, in Python, on the
  calling thread. It needs datatrove 0.10.1 and the segmenter installed, and
  `pyroid.PyroidError` says what is missing.
- The rest runs in Rust on those words, so the decisions are datatrove's. That is the filter
  math, or `simplify_text`, shingling, MinHash, bucketing and clustering. `threads=` applies to
  the Rust part only.
- The filters only tokenize the documents that reach a word-based check.
- A `C4Quality` step (which splits sentences with the tokenizer) runs as datatrove's own
  `C4QualityFilter`. In `apply`, later steps then see (and tokenize again) its rewritten text,
  as in a datatrove pipeline, and consecutive steps with the same tokenizer share one
  tokenization per document.
- Each function emits one [`pyroid.FallbackWarning`](../concepts/fallbacks.md) per language
  (C4 steps their own; `MinHashDedup` once, when the index is made).
- It also works for native codes, where it uses datatrove's tokenizer. That is slower and no
  more exact, so it is only needed for the others.

Sentence dedup has no Python-tokenizer path. A code whose datatrove tokenizer pyroid does not
port raises `NotImplementedError`, unless no tokenizer is needed (`split_sentences=False` and
`min_doc_words`, `min_num_sentences` and `min_words_to_remove_span` all `<= 0`). The datatrove
sentence-dedup steps in [`pyroid.integrations.datatrove`](datatrove.md) run datatrove's own code
for such codes, with a `FallbackWarning`.

!!! note "The datatrove drop-in steps"
    In `pyroid.integrations.datatrove`, the Gopher, C4, FineWeb, `MinhashDedupSignature` and
    sentence-dedup steps run in Rust for every native code. For the other codes they run
    datatrove's own code, with a `FallbackWarning`. The drop-ins have no `python_tokenizer=True`
    path. See [the datatrove guide](datatrove.md#languages).

## Native languages: German and French

FineWeb-2 sets thresholds and stop words per language; pass them as you would to datatrove:

```python
import pyroid.dedup as D
import pyroid.filters as F

de = [
    "Die Stadt liegt am Ufer des Flusses und war seit dem Mittelalter wichtig für Kaufleute aus "
    "vielen Ländern. Heute leben hier etwa achtzigtausend Menschen. Die Altstadt mit ihren "
    "engen Gassen und alten Häusern zieht jedes Jahr viele Besucher an. Im Sommer finden auf "
    "dem Marktplatz Konzerte statt, und im Winter gibt es einen kleinen Weihnachtsmarkt. Der "
    "Bahnhof liegt nicht weit vom Zentrum entfernt, und die Züge fahren regelmäßig in die "
    "größeren Städte der Region. Wer die Umgebung erkunden möchte, kann Fahrräder mieten und "
    "den Wegen am Fluss folgen.",
    "Jetzt kaufen! Jetzt kaufen! Jetzt kaufen!",
]
stop_words = ["der", "die", "das", "und", "mit", "nicht", "es", "in", "zu", "den"]
F.gopher_quality_reasons(de, language="deu_Latn", stop_words=stop_words)
# [None, 'gopher_short_doc']

fr = [
    "Le village se trouve au pied de la montagne, à quelques kilomètres de la frontière. "
    "On y vient pour les randonnées en été et pour le ski en hiver. L'église date du "
    "douzième siècle et le marché a lieu chaque samedi matin devant la mairie. "
    "Les habitants parlent volontiers de l'histoire de leur région, qui a longtemps vécu "
    "de l'agriculture et de l'élevage. Aujourd'hui, le tourisme est devenu la principale "
    "source de revenus, mais les fermes n'ont pas disparu.",
]
result = F.apply(
    fr,
    [
        F.GopherRepetition(language="fra_Latn"),
        F.FineWebQuality(language="fra_Latn"),
        F.GopherQuality(language="fra_Latn", stop_words=["le", "la", "de", "et", "les"]),
    ],
)
result.keep  # [True]

near = [de[0], de[0].replace("achtzigtausend", "neunzigtausend"), fr[0]]
D.minhash_dedup(near, language="deu_Latn")  # [True, False, True]
```

datatrove's `GopherQualityFilter(language="deu_Latn", stop_words=stop_words)` gives the same two
reasons, and the same French chain keeps the document. The MinHash signatures are datatrove's.

## Chinese

`cmn_Hani` and the other `SpaCyTokenizer("zh")` codes are native. datatrove segments them with
jieba through spaCy, and pyroid runs the same segmentation in Rust, so neither datatrove nor
jieba is needed:

```python
import pyroid.dedup as D
import pyroid.filters as F

zh = [
    "这座城市位于河边，自古以来就是重要的商业中心。今天，这里大约有八万人口。"
    "老城区的街道很窄，房子也很古老，每年都吸引很多游客。夏天，广场上有音乐会；"
    "冬天，有一个小小的圣诞市场。火车站离市中心不远，火车定期开往附近的大城市。"
    "想要了解周边地区的人，可以租一辆自行车，沿着河边的小路骑行。",
]
zh.append(zh[0].replace("八万", "九万"))  # a near-duplicate

F.fineweb_quality(zh, language="cmn_Hani")  # [True, True]
D.minhash_dedup(zh, language="cmn_Hani")  # [True, False]
```

The words are those of jieba 0.42.1 with its default dictionary, as datatrove gets them from an
unmodified jieba. Words added with `pyroid.jieba.add_word` or `load_userdict` do not change them.
To tokenize with a customised jieba, pass `python_tokenizer=True` and change jieba itself.

## Japanese

`jpn_Jpan` and the other `SpaCyTokenizer("ja")` codes (`ja`, `ja_Jpan`, `jpn`) are native when a
SudachiDict dictionary is installed. datatrove segments them with SudachiPy in split mode A,
through spaCy. pyroid ports SudachiPy 0.7.0's tokenizer and datatrove's handling around it
(40,000-byte chunks, the alignment of morphemes with the text, the sentencizer) to Rust, so
neither datatrove, spaCy nor SudachiPy is needed, only the dictionary:

```console
$ pip install sudachidict-core
```

```python
import pyroid.dedup as D
import pyroid.filters as F

F.languages()["jpn_Jpan"]  # Language(tokenizer='SpaCyTokenizer', arg='ja', native=True)

ja = [
    "この町は川のほとりにあり、昔から商業の中心として栄えてきました。今では約八万人が暮らしています。"
    "旧市街の通りは狭く、古い家が並び、毎年多くの観光客が訪れます。夏には広場で音楽会が開かれ、"
    "冬には小さなクリスマスマーケットが立ちます。駅は中心部から近く、近くの大きな町へ電車が定期的に走っています。"
    "周辺を見て回りたい人は、自転車を借りて川沿いの小道を走ることができます。",
]
ja.append(ja[0].replace("八万", "九万"))  # a near-duplicate

F.fineweb_quality(ja, language="jpn_Jpan")  # [True, True]
D.minhash_dedup(ja, language="jpn_Jpan")  # [True, False]
```

pyroid does not ship the dictionary: SudachiDict core's `system.dic` is 202 MB (the
sudachidict-core wheel is 77 MB). pyroid reads the `system.dic` of the installed
`sudachidict-core`, version 20261015 or newer (the SudachiPy 0.7 format), and finds the package
without importing it or SudachiPy. If `PYROID_SUDACHI_DICT` is set, the file it names is read
instead. The dictionary is read once per process, into memory (about 200 MB), the first time a
filter pipeline, MinHash or sentence dedup is built for a Japanese code. A dictionary in an older
format raises `ValueError`, and a path that cannot be read `OSError`. Parity was checked with
sudachidict-core 20261015 ([Parity](#parity)).

Without a dictionary these codes are not native: they raise `NotImplementedError`, whose message
says to install sudachidict-core, unless you pass `python_tokenizer=True` as in the next section.

## The Python tokenizer: Vietnamese with pyvi

`vie_Latn` is `SpaCyTokenizer("vi")`, which segments with pyvi. Without
`python_tokenizer=True`, the call says what to do:

```text
NotImplementedError: pyroid has no native word tokenizer for language 'vie_Latn': datatrove
tokenizes it with SpaCyTokenizer("vi") (spaCy's Vietnamese tokenizer with pyvi). Pass
python_tokenizer=True to tokenize it with datatrove's own tokenizer in Python, on the calling
thread (needs `pip install datatrove==0.10.1 regex spacy==3.8.18 pyvi`); the filter math still
runs in Rust.
```

With datatrove 0.10.1, spaCy 3.8.18 and pyvi installed:

```python
vi = [
    "Thị trấn nằm bên bờ sông và từ thời Trung cổ đã là nơi buôn bán quan trọng của các thương "
    "nhân đến từ nhiều nước. Ngày nay ở đây có khoảng tám mươi nghìn người sinh sống. Khu phố cổ "
    "với những con đường hẹp và những ngôi nhà cũ thu hút rất nhiều du khách mỗi năm. Vào mùa hè, "
    "trên quảng trường có các buổi hòa nhạc, còn vào mùa đông có một chợ Giáng sinh nhỏ. Nhà ga "
    "không xa trung tâm, và tàu hỏa chạy thường xuyên đến các thành phố lớn trong vùng. Ai muốn "
    "khám phá vùng xung quanh có thể thuê xe đạp và đi theo những con đường nhỏ dọc bờ sông.",
]
vi.append(vi[0].replace("dọc bờ sông", "dọc bờ hồ"))  # a near-duplicate

F.fineweb_quality(vi, language="vie_Latn", python_tokenizer=True)  # [True, True]
D.minhash_dedup(vi, language="vie_Latn", python_tokenizer=True)  # [True, False]
```

Each call warns once:

```text
FallbackWarning: pyroid.filters.fineweb_quality: tokenizing language 'vie_Latn' with datatrove's
SpaCyTokenizer("vi") (spaCy's Vietnamese tokenizer with pyvi) in Python, one document at a time on
the calling thread; the filter math runs in Rust. Expect the tokenizer's speed, not pyroid's.
This warning is shown once.
```

An unknown code raises datatrove's error before any document is read:

```python
F.gopher_quality(vi, language="und_Zzzz")
# ValueError: Language 'und_Zzzz' doesn't have a tokenizer assigned. Pass in a WordTokenizer
# directly or update tokenizer_assignment.csv
```

## Parity

Every check below found 0 differences against datatrove 0.10.1 (spaCy 3.8.18, and SudachiPy 0.7.0
with sudachidict-core 20261015 for Japanese; CPython 3.13 for the live checks):

| Check | Rows |
|---|--:|
| Tokenizer replay, committed goldens: English (3,559) and 61 per-language files (57 spaCy languages, whitespace, IndicNLP hi/ur/bn) | 43,146 |
| … and, not committed: whole FineWeb-2 slices (ar, de, fr, ru, es, hi), 10k fuzz rows per tokenizer, a 57-language FineWeb-2 smoke set | 662,530 |
| IndicNLP replay, with 8k fuzz rows per CPython 3.11–3.15 (each version matches its own) | 118,690 |
| Whitespace tokenizer replay | 118,990 |
| Filter suites, 8 FineWeb-2 slices × 8 cases: FineWeb-2's 3 filters with that language's thresholds and stop words, and datatrove's defaults including C4 (Japanese through `python_tokenizer=True`) | 16,000 decisions |
| MinHash with FineWeb-2's configuration (xxhash/64) and the language: signatures and removed set, 8 slices × 300 documents with near-duplicates injected | 2,400 + 2,400 |
| Hypothesis against live datatrove: script-specific text, 22 native languages and jieba, SudachiPy, pyvi and PyThaiNLP | 92 tests × 300 examples |
| Chinese, native: the `cmn_Hani` slice's 8 filter cases, and its MinHash signatures and removed set | 2,000 decisions + 300 + 300 |
| Chinese word segmentation, [`pyroid.jieba`](jieba.md#parity) against jieba 0.42.1 | 580,793 comparisons |
| Japanese, native: the `jpn_Jpan` slice's 8 filter cases, and its MinHash signatures and removed set | 2,000 decisions + 300 + 300 |
| Japanese tokenizer replay, committed recordings: SudachiPy's morphemes for 4,567 texts (special cases, FineWeb-2 lines, fuzz, a code-point sample), datatrove's words, spans and sentences for 4,205 texts (54 documents over 40,000 bytes), 1,911 `chunk_text_on_bytes` cases, the spaCy tag of all 1,558 POS | 12,241 |
| … and, not committed: every code point, the FineWeb-2 `jpn_Jpan` test slice (7,809 lines, 250 documents) and the first 20,000 training documents (32.5 M characters: 610,624 lines with 18,879,619 morphemes, and the documents' words and spans) | 777,691 |

Japanese also has 12 Hypothesis properties against live datatrove
(`python/tests/test_filters_japanese.py`): 628 examples per run, and 11 of them also at 300
examples each.

Sentence dedup has FineWeb-2 parity cases on six slices (`deu_Latn`, `fra_Latn`, `rus_Cyrl`,
`spa_Latn`, `arb_Arab`, `hin_Deva`), for sentences and lines. The Stanza, Kiwi, Khmer, Lao,
Tibetan and Burmese segmenters were not installed in these runs, so their
`python_tokenizer=True` path is untested. See [Compatibility](../compatibility.md#filters).

Quirks reproduced as datatrove has them:

- In C4 for Hindi and Bengali, indic-nlp re-joins sentences with `" "`, so datatrove's
  `text.index` raises `ValueError` on some documents (9 of 250 real `hin_Deva` documents).
  pyroid raises the same.
- Urdu sentences (C4 only) follow `datatrove[multilingual]`, where indic-nlp imports urduhack.
  With plain datatrove and indic-nlp-library, datatrove raises `ModuleNotFoundError` for every
  Urdu sentence call.

spaCy is quadratic on long glued runs of characters (9.6 s for one 4.7 kB text); pyroid is linear
there.

## Speed

These timings come from a Docker/OrbStack VM on Apple Silicon, indicative. They cover FineWeb-2's
filter chain (Gopher repetition, FineWeb quality and Gopher quality, with each language's
settings) and its MinHash signatures, on 1,000 documents per language (each FineWeb-2 slice four
times). "8" is 8 threads for pyroid and a warmed 8-process `Pool` for datatrove. Medians of 5
runs (datatrove: 2), with every timed document checked first for the same decisions and
signatures. Release wheel, CPython 3.13, load average 6 to 16:

| Language | Tokenizer | Filters, 1 thread (µs per document, datatrove → pyroid) | Speedup: filters / MinHash, 1 thread | 8 threads vs 8 processes |
|---|---|--:|--:|--:|
| `deu_Latn` | native | 8,342 → 122 | 68× / 53× | 54× / 55× |
| `fra_Latn` | native | 16,853 → 195 | 86× / 64× | 15×\* / 42× |
| `spa_Latn` | native | 14,553 → 218 | 67× / 51× | 28× / 31× |
| `rus_Cyrl` | native | 16,478 → 457 | 36× / 36× | 55× / 42× |
| `arb_Arab` | native | 8,225 → 145 | 57× / 49× | 42× / 38× |
| `hin_Deva` | native (IndicNLP) | 2,514 → 126 | 20× / 10× | 15× / 9× |
| `cmn_Hani` | native (jieba)† | 10,628 → 242 | 44× / 44× | 43× / 42× |
| `jpn_Jpan` | native (SudachiPy port)‡ | 11,210 → 1,444 | 7.8× / 6.4× | 9.2× / 6.9× |

\* An outlier: pyroid with 8 threads was slower than with 1 thread in this run (246 µs per
document) while the load rose. It needs a rerun before it is quoted.

† Measured on 2026-11-07, after the jieba port, at load average 1 to 2. Before it, `cmn_Hani` went
through `python_tokenizer=True`, at 1.2× / 0.9× on one thread and 0.4× with 8.

‡ Measured on 2026-11-09, after the SudachiPy port, at load average 0.6 to 1.9. Before it, `jpn_Jpan`
went through `python_tokenizer=True`, at 1.2× / 0.9× on one thread and 0.3× / 0.2× with 8.
Japanese gains less than the other native languages because SudachiPy's lattice dominates on both
sides: pyroid's whole MinHash (1,151 µs per document, one thread) takes less time than SudachiPy's
own `tokenize()` call on the same chunks (1,583 µs).

The native languages run the filters 36–86× faster than datatrove on one thread (Hindi: 20×,
Japanese: 7.8×).
The Python-tokenizer path runs at the reference tokenizer's speed (about 1×, measured for Chinese
and Japanese before their ports) and does not scale with threads, so datatrove's 8 processes beat
it (0.2–0.4×). Adding the languages did not slow
English down: old and new builds measured within noise on 5,000 FineWeb documents. See
[Benchmarks](../benchmarks.md#other-modules-languages-and-integrations).

## Binary size

The language tables add 1.11 MB to the release extension module (fat LTO, stripped, aarch64). It
grew from 3.29 to 4.40 MB, or from 1.20 to 1.46 MB gzipped. Chinese adds
[`pyroid.jieba`](jieba.md)'s dictionary and HMM tables: 3.0 MB more (5.58 to 8.60 MB), and 1.5 MB
more in the wheel. Japanese adds the SudachiPy port and SudachiPy's resource files (`char.def`,
`unk.def`, `rewrite.def`, Apache-2.0), but not the dictionary: the release extension module
(aarch64, abi3) grew from 9.19 to 9.32 MB, and the wheel by 65 KB.

API: [`pyroid.filters` reference](../reference/filters.md),
[`pyroid.dedup` reference](../reference/dedup.md).
