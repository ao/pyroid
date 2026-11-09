# `pyroid.filters`: datatrove's filters

`pyroid.filters` ports these [datatrove](https://github.com/huggingface/datatrove) 0.10.1 steps to
Rust (datatrove is © Hugging Face, Apache-2.0):

- the quality filters `GopherQualityFilter`, `GopherRepetitionFilter`, `C4QualityFilter` and
  `FineWebQualityFilter`;
- `C4BadWordsFilter` and `URLFilter`;
- the formatters `PIIFormatter` and `SymbolLinesFormatter`.

They keep and drop the same documents as datatrove and give the same `filter_reason` strings.
After C4's line removal, PII replacement and symbol-line removal the text is the same too
([parity](../compatibility.md#filters)).

The hard part is the words. datatrove tokenizes each language with its own tokenizer; for English
that is spaCy 3.8.18's `spacy.blank("en")` plus the sentencizer. Every word count, ratio and
n-gram in the quality filters depends on those tokens. pyroid includes exact ports of datatrove's
rule-based tokenizers. For 57 languages it ports spaCy's prefix, suffix, infix and URL rules, its
tokenizer exceptions, the special-case matcher and the sentencizer. It also ports the Indic NLP
and whitespace tokenizers, spaCy's Chinese tokenizer with jieba ([`pyroid.jieba`](jieba.md)), and
datatrove's Japanese tokenizer with SudachiPy, which reads an installed SudachiDict dictionary.
Together they cover 4,013 of datatrove's 4,343 language codes ([Languages](#languages)).

## From a datatrove pipeline

datatrove filters are pipeline steps that read `Document` objects. pyroid's are functions over a
batch of texts that return masks, so they fit wherever you have the texts: a list, a Parquet
column, a `datasets` batch.

```diff
-from datatrove.pipeline.filters import GopherQualityFilter
-pipeline = [reader, GopherQualityFilter(min_doc_words=100), writer]
+import pyroid.filters as F
+keep = F.gopher_quality(texts, min_doc_words=100)   # one bool per text: datatrove's decision
```

To keep the datatrove pipeline and swap its steps for pyroid-backed ones instead, see
[datatrove steps](datatrove.md).

To run several filters, chain them in one call with `apply`. It runs them in datatrove's pipeline
order: a document dropped by one filter is not seen by the next, later filters see C4's rewritten
text, and steps that see the same text tokenize it once.

```python
import pyroid.filters as F

result = F.apply(
    texts,
    [
        F.GopherRepetition(),
        F.GopherQuality(),
        F.C4Quality(filter_no_terminal_punct=False),  # FineWeb's C4 settings
        F.FineWebQuality(),
    ],
)
result.keep  # keep mask, e.g. [True, False, ...]
result.reason  # datatrove's filter_reason, None when kept, e.g. [None, 'dup_line_frac', ...]
result.text  # each document's text after the chain (C4 removes lines)
kept = [t for t, k in zip(result.text, result.keep) if k]
```

You can pass datatrove's own filter objects instead of pyroid's configs, and their settings are
read from them:

```python
from datatrove.pipeline.filters import GopherQualityFilter

F.apply(texts, [GopherQualityFilter(min_doc_words=100)])
```

## One filter at a time

Each filter has three functions, all with datatrove's parameter names and defaults. Pass the
parameters as keyword arguments or as a config object such as `F.GopherQuality(...)`:

| Function | Returns, per document |
|---|---|
| `gopher_quality(texts, **params)` | `True` if datatrove keeps it |
| `gopher_quality_reasons(texts, **params)` | datatrove's drop reason, or `None` when kept |
| `gopher_repetition`, `gopher_repetition_reasons` | the same, for `GopherRepetitionFilter` |
| `c4_quality`, `c4_quality_reasons`, `c4_lines` | the same for `C4QualityFilter`; `c4_lines` returns the text after C4's line removal |
| `fineweb_quality`, `fineweb_quality_reasons` | the same, for `FineWebQualityFilter` |

```python
F.gopher_quality(texts, min_doc_words=100)
F.gopher_repetition_reasons(texts, dup_line_frac=0.25)
F.c4_lines(texts, F.C4Quality(filter_no_terminal_punct=False))
```

A falsy threshold (`None`, `0`) disables its check, as in datatrove. Bad parameter types and
values are checked when the call starts, before any document is filtered.

## Arrow, `datasets` and threads

Predicates return a mask in the input's kind. Arrow input gives a
[`pyroid.Column`](../reference/pyroid.md#pyroid.Column), and nulls stay null:

```python
import pyarrow as pa
import pyarrow.parquet as pq
import pyroid.filters as F

table = pq.read_table("fineweb.parquet")
keep = F.fineweb_quality(table["text"], threads=8)  # pyroid.Column of bools
table = table.filter(pa.chunked_array(keep))
```

With Hugging Face `datasets`:

```python
ds = ds.filter(lambda b: F.gopher_quality(b["text"]), batched=True, batch_size=10_000)
```

## Languages

`language` (a field of every quality-filter config) takes any of datatrove's 4,343 language
codes, as datatrove's `language=` does: `"eng"` (the default), `"deu_Latn"`, `"deu"`, `"de"`, ...

- 4,013 codes are native, tokenized in Rust on every thread. They are the 57 spaCy languages
  datatrove tokenizes with a rule-based `spacy.blank(code)` (with `xx`, spaCy's multi-language
  tokenizer), Chinese (spaCy's Chinese tokenizer with jieba), Japanese (datatrove's tokenizer
  with SudachiPy), the Indic NLP languages and datatrove's whitespace tokenizer, and they make up
  95.6% of FineWeb-2's bytes. Japanese needs a SudachiDict dictionary:
  `pip install sudachidict-core`, or the `system.dic` that `PYROID_SUDACHI_DICT` names
  ([Japanese](languages.md#japanese)). Without one, 4,009 codes (88.4%) are native.
- The other 330 (Vietnamese with pyvi, Stanza, Thai, Kiwi, ...) raise `NotImplementedError` unless you pass `python_tokenizer=True`. Then
  datatrove's own tokenizer splits each document, in Python on the calling thread, and the filter
  math runs in Rust, with a [`FallbackWarning`](../concepts/fallbacks.md).
- A code datatrove does not know raises datatrove's `ValueError` when the call starts.

`F.languages()` lists every code with its tokenizer and whether pyroid's is native.
`F.english_languages()` lists the codes datatrove tokenizes as English. `language` may also be a
word tokenizer object: anything with `word_tokenize(text)`, and `span_tokenize(text)` for C4.
Python string semantics (`isalpha`, `isspace`, `splitlines`, `lower`, `re`'s `\w` and `\d`)
follow the running interpreter's Unicode version, as datatrove's do.

The language table, how to choose, examples, parity and per-language speed are in
[Languages](languages.md).

## PII

`pii_replace` is datatrove's `PIIFormatter`. Every match of its email regex becomes the next
email replacement. Then every match of its IPv4 regex that `ipaddress.ip_address` accepts becomes
the next IP replacement; by default only addresses whose `is_global` is true are replaced.

```python
import pyroid.filters as F

docs = [
    "Write to jane.doe@example.com or call the office.",
    "Server at 8.8.8.8, router at 192.168.0.1.",
    "Nothing to hide here.",
]
F.pii_replace(docs)
# ['Write to email@example.com or call the office.',
#  'Server at 22.214.171.124, router at 192.168.0.1.',
#  'Nothing to hide here.']
F.pii_replace(
    docs, email_replacement="<EMAIL>", ip_replacement="<IP>", only_remove_public_ips=False
)
# ['Write to <EMAIL> or call the office.',
#  'Server at <IP>, router at <IP>.',
#  'Nothing to hide here.']
```

datatrove's default replacements (`F.PII_EMAIL_REPLACEMENT`, `F.PII_IP_REPLACEMENT`) are tuples
used in turn, and the position in each tuple carries over from one document to the next. Each
`pii_replace` call starts at the first one, so one call over a whole corpus gives datatrove's
output. To process a corpus in batches, keep one `PIIReplacer`, whose counters carry over across
calls:

```python
pii = F.PIIReplacer()
pii(["Mail a@b.co."])  # ['Mail email@example.com.']
pii(["Mail c@d.org or e@f.net."])  # ['Mail firstname.lastname@example.org or email@example.com.']
pii.email_index  # 1: the next email replacement (datatrove's counter; assignable)
```

The matching and replacing run in Rust, in parallel. `ipaddress` is asked about each distinct
address once, on the calling thread, so `is_global` follows the running interpreter's tables, as
in datatrove. Documents with nothing to replace come back as the original objects.

## C4 bad words

`c4_badwords` is datatrove's `C4BadWordsFilter`. A document is dropped when its `text.lower()`
contains a word of its language's list, between non-word characters or the text's ends. Japanese,
Thai and Chinese words match anywhere. `languages=` gives each document's language, as datatrove
reads `doc.metadata["language"]`: one code for all, or one per document. The default is
`default_language="en"`:

```python
import pyroid.filters as F

texts = ["A recipe for apple pie.", "Nude beaches of the coast.", "Er schwamm nackt im See."]
F.c4_badwords(texts, languages=["en", "en", "de"])  # [True, False, False]
F.c4_badwords_reasons(texts, languages=["en", "en", "de"])
# [None, 'document_removed_with_badwords', 'document_removed_with_badwords']
F.c4_badwords(texts)  # [True, False, True]: every document is "en"
F.c4_badwords(texts, words={"en": ["recipe"]})  # [False, True, True]: your own list for "en"
```

The word lists are the [LDNOOBW](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words)
lists (CC BY 4.0) at the two commits datatrove pins, for the 28 languages in
`F.C4_BADWORDS_LANGUAGES`. pyroid does not ship them. Like datatrove, it downloads a list the
first time it is needed and checks its sha256. They are kept under `c4_badwords/` in
`$PYROID_CACHE_DIR`, else `$XDG_CACHE_HOME/pyroid`, else `~/.cache/pyroid`.

`F.load_c4_badwords("de")` returns one language's set (`download=False` raises
`FileNotFoundError` instead of fetching), and `words=` replaces the lists for the languages it
names. A document language with no list raises datatrove's `ValueError` before any document is
filtered (`fail_on_missing_language=True`, the default). With `False`, its documents are kept.

With `keep_fraction > 0`, a flagged document is kept anyway when the next `uniform()` draw of
numpy's `default_rng(seed)` is below it. pyroid reproduces numpy's PCG64 stream bit for bit, one
draw per flagged document in document order. `c4_badwords` starts a fresh generator per call. A
`BadWordsFilter` continues its stream across calls, as one datatrove step does over a corpus:

```python
f = F.BadWordsFilter(keep_fraction=0.5, seed=42)
batch = ["nude 1", "nude 2", "nude 3", "nude 4", "fine"]
f(batch).keep  # [False, True, False, False, True]
f.keep(batch)  # [True, False, False, False, True]: the draws continue
f.found(batch)  # [True, True, True, True, False]: matches only, no draws
```

These are the decisions datatrove's `C4BadWordsFilter(keep_fraction=0.5, seed=42)` makes on the
same ten documents.

## Symbol lines

`symbol_lines` is datatrove's `SymbolLinesFormatter`. Over `text.splitlines()`, a non-blank line
made only of `symbols_to_remove` and spaces is dropped, and a run of such lines becomes one
`replace_char` (if it is not empty). The kept lines are joined with `"\n"`. The default symbols
are datatrove's `PUNCTUATION_SET`. FineWeb-2 removes `|` lines and puts a newline in their place:

```python
import pyroid.filters as F

text = "Title\n|||\n| |\nBody text\n---\nEnd"
F.symbol_lines(text)  # 'Title\nBody text\nEnd'
F.symbol_lines(text, symbols_to_remove=["|"], replace_char="\n")  # FineWeb-2's settings
# 'Title\n\n\nBody text\n---\nEnd'
```

## URL filter

`URLFilter` is datatrove's `URLFilter`: the same decisions and drop reasons for each URL (the
`doc.metadata["url"]` datatrove reads). Each URL is checked in datatrove's order, and the first
hit is the reason:

1. `"domain"`: its registered domain is blocked;
2. `"subdomain"`: its full host name is blocked;
3. `"url"`: the URL itself is blocked;
4. `"hard_blacklisted"`: a word of the URL (split on `[^a-zA-Z0-9]+`) is a banned word;
5. `"soft_blacklisted"`: at least `soft_word_threshold` distinct words are soft-banned;
6. `"blacklisted_subword"`: a banned subword is a substring of the normalized URL.

```python
import pyroid.filters as F

urls = [
    "https://www.example.com/about",
    "https://ads.example.net/x",
    "https://shop.example.org/casino-bonus",
    "https://docs.python.org/3/",
]
uf = F.URLFilter(
    use_integrated_lists=False,
    extra_domains=["example.com", "ads.example.net"],
    banned_words=["casino"],
)
uf.keep(urls)  # [False, False, False, True]
uf.reasons(urls)  # ['domain', 'subdomain', 'hard_blacklisted', None]
uf.domains(urls)[0]  # ['example.com', 'example.net', 'example.org', 'python.org']
F.url_filter_reasons(urls, uf)  # the same reasons, from the built filter
```

Domains come from tldextract 5.4.0's `extract_str`, ported to Rust, with the suffix lists of a
`tldextract.TLDExtract`. Pass `tld_extractor=`; the default is a new `TLDExtract()`, as datatrove
builds. tldextract must be installed.

By default (`use_integrated_lists=True`) the filter also uses datatrove's own block lists, so
datatrove must be installed. pyroid does not ship them, because they are not under pyroid's
licence. `F.datatrove_url_lists()` extracts them from datatrove's package into datatrove's cache,
as datatrove does, so both share one copy. Build the filter once and reuse it: it holds 4.56
million blocked domains (about 200 MB) and takes a second or two to build.

```python
full = F.URLFilter()  # datatrove's lists (needs datatrove and fasteners installed)
full.num_domains, full.num_urls  # (4558940, 19587)
```

`url_filter(urls, **params)` and `url_filter_reasons` build a filter from a config, or take a
built one. The last two filters built from configs are cached. Two rare inputs run tldextract's
own Python code, with a `FallbackWarning`: URLs with lone surrogates, and `xn--` labels when the
installed `idna` lacks the canonical A-label check.

## Differences from datatrove

From the module's documentation (`pyroid/filters.pyi`):

- A `top_n_grams` / `dup_n_grams` size below 1 raises `ValueError` (datatrove loops forever on
  `dup_n_grams` sizes below 1).
- Values are checked when the call starts. A non-number threshold raises `TypeError`, and an
  unknown `language` `ValueError`, before any document is filtered; datatrove raises the latter
  at the first document that needs words.
- A failing document raises for the whole batch, and the exception names the document.
- `exclusion_writer` is not supported (use the `*_reasons` functions).
- A document with a token that is exactly `IS_ALPHA` (e.g. in `"IS_ALPHA."`, not in
  `"xIS_ALPHA"`) is tokenized like any other text, as datatrove does the first time in a
  process. After that, spaCy fails on such documents (its string store maps `IS_ALPHA` to id 1,
  which is its hash map's deleted-key marker), and datatrove re-tokenizes
  `text.replace("IS_ALPHA", "")` instead, or raises if that still has the token. So datatrove's
  words for these documents depend on what it saw before; pyroid's do not.

For the newer steps:

- Errors are raised before a batch is processed, where datatrove raises mid-stream: a bad-words
  language with no list (with `fail_on_missing_language`), an empty URL (`ValueError`, where
  datatrove's `URLFilter` fails an assertion), and an empty replacement tuple that a match needs
  (`IndexError`).
- datatrove's quirks are kept. In `URLFilter`, a whitespace-only list entry puts `""` among the
  blocked domains, which drops every URL without a registered domain as `"domain"`, and an
  empty-string entry raises `IndexError`. In `PIIFormatter`, IP addresses with leading zeros
  never parse, so they are never replaced.
- The datatrove drop-in steps have their own notes: see [datatrove steps](datatrove.md).

Also: on a Python newer than Unicode 17.0, pyroid warns once and uses its 17.0 tables. Filtering
runs in Rust except on the paths that emit a `FallbackWarning`: a Python tokenizer
(`python_tokenizer=True`, or a tokenizer object as `language`) and the two rare URL inputs. Other
datatrove filters (`C4ParagraphFilter`, `UnigramLogProbFilter`, ...) are not ported. For
`UnigramLogProbFilter`, nothing grants rights to its word-frequency data.

## Speed

On an AWS c7i.4xlarge (Intel Xeon Platinum 8488C), over the first 1,000 FineWeb documents, single
filters on one thread were 35–69× faster than datatrove 0.10.1. GopherQuality, for example, took
99.9 µs per document against 4,615 µs. With 8 threads the filters took 13.2–26.4 µs per document.
pyroid ran on free-threaded CPython 3.15t and datatrove on CPython 3.13. On a c8g.4xlarge (AWS
Graviton4) the range was 41–83×. See [Benchmarks](../benchmarks.md#per-stage).

The rest of this section comes from earlier runs on a Docker/OrbStack VM on Apple Silicon,
indicative.

FineWeb's four filters chained through `apply` measured about 86× faster on one core in the
module's preliminary run, at 0.28 against 23.9 ms per document
([details](../benchmarks.md#pyroidfilters)).

Beyond English, FineWeb-2's filter chain ran 36–86× faster than datatrove on one thread for the
native languages measured (Hindi: 20×), 44× for Chinese and 7.8× for Japanese, where SudachiPy's
lattice dominates ([Languages](languages.md#speed)).

The table times the newer steps on the FineWeb dev slice (10,000 documents). datatrove's step ran
alone in one process and in a warmed `multiprocessing.Pool(8)`, and pyroid's function at 1 and 8
threads. Release wheel, CPython 3.13, `taskset -c 0-7`; medians of 5 for pyroid and 2 for
datatrove; load average about 8.

| Step | datatrove, 1 process | 8 processes | pyroid, 1 thread | 8 threads | Speedup: 1 thread vs 1 process / 8 threads vs 8 processes |
|---|--:|--:|--:|--:|--:|
| `pii_replace` (`PIIFormatter`) | 2.885 s | 0.595 s | 0.032 s | 0.019 s | 89× / 31× |
| `c4_badwords` (`C4BadWordsFilter`) | 6.968 s | 1.312 s | 0.305 s | 0.053 s | 23× / 25× |
| `symbol_lines` (`SymbolLinesFormatter`, FineWeb-2's settings) | 0.145 s | 0.198 s | 0.028 s | 0.008 s | 5.2× / 25× |

`URLFilter` was timed on the 10,000 dev URLs, at a load average of about 22. On one thread it
checked 912k URLs/s, 31× datatrove's `filter()` loop (29.4k URLs/s). With 8 threads it reached
1.72M URLs/s, 59× datatrove's single process. That run was limited by contention, and datatrove
at 8 processes was not measured. Building the filter with the integrated lists took 1.44 s and
198 MB more memory, against 2.66 s and 587 MB per process for datatrove. See
[Benchmarks](../benchmarks.md#other-modules-languages-and-integrations).

API: [`pyroid.filters` reference](../reference/filters.md).
