# `pyroid.trafilatura`: drop-in Trafilatura

`pyroid.trafilatura` is a Rust port of the text extraction of
[Trafilatura](https://github.com/adbar/trafilatura) by Adrien Barbaresi (Apache-2.0), releases 2.3.1
(the default) and 1.11.0 (the one `datatrove[processing]` 0.10.1 installs, [below](#trafilatura-1110)).
Trafilatura turns a web page into its main text, without the navigation, footers, ads and other
boilerplate; datatrove's `Trafilatura` step, and so FineWeb, use it. The whole path is ported,
including the parts Trafilatura delegates: libxml2 2.14.6's HTML parser as lxml 6.1.3 drives it,
the readability fallback Trafilatura vendors, and jusText 3.0.2. The text comes out the same, byte
for byte, as Trafilatura with lxml 6.1.3 and jusText 3.0.2 ([parity](#parity)), 7 to 11 times
faster on one thread, and in parallel without the GIL.

```bash
pip install pyroid
```

pyroid has no dependencies. trafilatura is only needed for the [delegated options](#delegated-options).

## Change one import

```diff
-import trafilatura
+import pyroid.trafilatura as trafilatura

 html = "<html><body><article><p>" + "Some words. " * 30 + "</p></article></body></html>"
 trafilatura.extract(html)                    # 'Some words. Some words. ...'
 trafilatura.extract(html, favor_precision=True, include_comments=False, deduplicate=True)
```

`extract`, `bare_extraction` and `extract_with_metadata` have trafilatura 2.3.1's parameters, in
the same order, with the same defaults. The one addition is the keyword-only `compat`
([versions](#versions)). Code you cannot edit can use [`pyroid.patch("trafilatura")`](#pyroidpatchtrafilatura).

## What runs in Rust

- Input: a `str` (lone surrogates included), or `bytes` that are valid UTF-8 and not compressed.
- `output_format="txt"` for `extract`, and `"python"` for `bare_extraction`.
- `fast` (and the deprecated `no_fallback`, with trafilatura's `DeprecationWarning`),
  `favor_precision`, `favor_recall`, `include_comments`, `include_tables`.
- `deduplicate`: `True`, or a [`pyroid.trafilatura.LRUCache`](#deduplication).
- `url`, `record_id`, `tei_validation` and `date_extraction_params`, which do not change the text
  output; `max_tree_size` raises trafilatura's `ValueError`.
- `config` and `settingsfile`: the size settings (`MIN_EXTRACTED_SIZE`, `MIN_OUTPUT_SIZE`,
  `MIN_OUTPUT_COMM_SIZE`, `MIN_DUPLCHECK_SIZE`, `MAX_REPETITIONS`, `MAX_TREE_SIZE`) are read as
  trafilatura reads them, with the same errors for bad values.
- The interpreter's Unicode version (CPython 3.11 to 3.15), as trafilatura's `str` methods and
  NFC normalization see it.

Other input types give `None`, as in trafilatura. Pages trafilatura rejects (too short, not
HTML, duplicates) give `None` too, and a page with a `&#` reference of more than 4,300 digits
raises CPython's `ValueError`, as trafilatura does.

## Delegated options

These go to the installed trafilatura, which must be the mode's release (2.3.1 by default, 1.11.0
with `compat="1.11.0"`, where `author_blacklist` and `extract_with_metadata` are not on this
list). The first call per function and reason emits a
[`pyroid.FallbackWarning`](../concepts/fallbacks.md); without trafilatura, or with another
version, the call raises [`pyroid.PyroidError`](../reference/pyroid.md#pyroid.PyroidError) and
says which:

- the other output formats (`xml`, `xmltei`, `json`, `csv`, `html`, `markdown`);
- `include_formatting=True`, `include_links=True`, `include_images=True`;
- metadata: `with_metadata`, `only_with_metadata`, a non-empty `url_blacklist` or
  `author_blacklist`, and `extract_with_metadata` (always delegated);
- `target_language` and `prune_xpath`;
- `options=` a `trafilatura.settings.Extractor`, and `deduplicate=` a
  `trafilatura.deduplication.LRUCache`;
- an lxml tree or an HTTP response as input, and bytes that are compressed or not UTF-8;
- a negative size setting in `config`, and a `sys.set_int_max_str_digits()` limit other than 4300.

```python
trafilatura.extract(html, output_format="xml")  # trafilatura's XML, and:
# FallbackWarning: pyroid.trafilatura.extract: output_format='xml' is not supported by the Rust
# port; delegating to trafilatura 2.3.1 (slow path). This warning is shown once.
```

A pyroid `LRUCache` (or the module cache, for `deduplicate=True`) is lent to a delegated call, so
a run that mixes native and delegated calls deduplicates as one trafilatura run would.

## Deduplication

`deduplicate=True` reads and updates one module-level cache, `pyroid.trafilatura.LRU_TEST`, of
`LRU_SIZE` (4,096) entries, as trafilatura's `trafilatura.deduplication.LRU_TEST`.
`reset_caches()` empties it, as `trafilatura.meta.reset_caches()` does. Pass a
`pyroid.trafilatura.LRUCache` instead to keep a run's state apart:

```python
import pyroid.trafilatura as trafilatura

cache = trafilatura.LRUCache(maxsize=4096)
texts = [trafilatura.extract(page, deduplicate=cache) for page in pages]
cache.maxsize, len(cache.cache)  # cache.cache is a read-only copy of the entries
```

`LRUCache` has trafilatura's methods (`get`, `put`, `increment`, `clear`) and is always true. It
is safe to share between threads: an extraction holds it for the whole page, so extractions
through one cache run one at a time. It pickles and copies with its entries. Keys are str without
lone surrogates.

## Many pages: `extract_batch`

`extract_batch` is pyroid's addition. It runs `extract` over a list or tuple of pages on pyroid's
thread pool, without the GIL, and returns the results in order:

```python
texts = trafilatura.extract_batch(pages, favor_precision=True, include_comments=False)
texts = trafilatura.extract_batch(pages, threads=4)  # at most 4 tasks at once
```

`extract_batch(pages, **kwargs)` always equals `[extract(page, **kwargs) for page in pages]`.
Without `deduplicate`, the pages are extracted in parallel. With `deduplicate=True` or an
`LRUCache`, they are parsed in parallel, but extracted one after the other in page order through
the cache, so the results, and the cache afterwards, are the loop's; such a batch gains less from
threads ([speed](#speed)). A batch with an item that is not a `str`, or with a delegated option
or `no_fallback`, runs that loop. An error raises for the whole batch, with a note naming the
page.

## `bare_extraction`

`bare_extraction` returns a `pyroid.trafilatura.Document` with trafilatura's `Document` fields:
`text`, `raw_text` (the same string), `comments` (with `include_comments`), `body` and
`commentsbody`, and `None` in the metadata fields, as trafilatura without metadata extraction.
When trafilatura 2.3.1 is installed, the class is a subclass of `trafilatura.settings.Document`
(trafilatura is imported the first time it is used; on free-threaded CPython that import
re-enables the GIL, as lxml does, while `extract` and `extract_batch` never import trafilatura or
lxml). `body` and `commentsbody` are built as lxml
elements from their XML when first read; without lxml they are the XML strings
(`etree.tostring(element, encoding="unicode")`). `as_dict()` and the deprecated `as_dict=True`
work as in trafilatura, and a `Document` pickles (its trees as XML).

## Versions

pyroid reproduces one combination of releases per mode, selected with `compat=`:

| `compat` | trafilatura | lxml | libxml2 | jusText | Also as resolved (no effect on the text) |
|---|---|---|---|---|---|
| `"2.3.1"` (the default) | 2.3.1 | 6.1.3 | 2.14.6 (bundled in lxml's wheels) | 3.0.2 | |
| `"1.11.0"` | 1.11.0 | 6.1.3 | 2.14.6 | 3.0.2 | lxml-html-clean 0.4.6, charset-normalizer 3.5.2, courlan 1.4.0, htmldate 1.11.0 |

The second row is what a fresh `pip install 'datatrove[processing]==0.10.1'` resolves.
`pyroid.trafilatura.TRAFILATURA_VERSIONS` lists the modes, `REPRODUCED_VERSIONS` the versions
behind each, and `DEFAULT_CONFIGS` each release's settings.cfg. Another version of trafilatura or
lxml can extract different text from the same page.

### Choosing a mode

| Use | `compat` |
|---|---|
| Direct calls, and code written for trafilatura 2.3.1 | `"2.3.1"`, the default |
| Output that must match a stock `datatrove[processing]==0.10.1` install, such as a FineWeb-style pipeline checked against stock datatrove | `"1.11.0"` |
| The [datatrove step](#the-datatrove-step) and [`pyroid.patch("trafilatura")`](#pyroidpatchtrafilatura) | leave it at `None`: they follow the installed trafilatura, so a stock datatrove install gets 1.11.0 |

## trafilatura 1.11.0

`compat="1.11.0"` runs trafilatura 1.11.0's extraction:

```python
import pyroid.trafilatura as trafilatura

trafilatura.extract(
    html, favor_precision=True, include_comments=False, deduplicate=True, compat="1.11.0"
)  # what datatrove 0.10.1's Trafilatura step gets
trafilatura.extract_batch(pages, no_fallback=True, compat="1.11.0")
```

The same options run natively as for 2.3.1, with 1.11.0's behaviour, and the delegated ones go to
an installed trafilatura 1.11.0. Where 1.11.0 differs from 2.3.1:

- It has no `fast`. Its `extract` takes `**kwargs` and ignores unknown keywords, so `fast=True`
  does nothing there, and neither does it here; `bare_extraction` raises 1.11.0's `TypeError`.
  The switch is `no_fallback`, without a `DeprecationWarning`.
- `max_tree_size` is a limit on the output tree, not an error. `author_blacklist` does not need
  metadata in 1.11.0, so it runs natively and has no effect on the text.
- `deduplicate` is a flag: any true value, a `pyroid.trafilatura.LRUCache` included, uses the
  module cache `LRU_TEST`, as 1.11.0 always uses its own.
- `config` wins over `settingsfile`, which is only read with `config=None`, and then on its own.
  The default `config` stands for 1.11.0's settings.cfg.
- `bare_extraction` raises 1.11.0's `AttributeError` (`'NoneType' object has no attribute
  'source'`) for a page that does not load, as 1.11.0 does when it is called directly, and its
  `commentsbody` is `None` without `include_comments`. Its `Document` is 1.11.0's (a subclass of
  it when 1.11.0 is installed). 1.11.0 keeps control characters, which no lxml element built from
  Python can hold: a tree with one stays its XML string.
- There is no `extract_with_metadata` in 1.11.0; `compat="1.11.0"` raises `ValueError` for it.

`bare_extraction` returns a dict unless `as_dict=False`, as 1.11.0's does, and `extract` ignores
keywords it does not know (the old `*_output` names raise 1.11.0's `NameError`). The functions
keep 2.3.1's parameter order, so pass 1.11.0's options by keyword.
[`pyroid.patch("trafilatura")`](#pyroidpatchtrafilatura) with trafilatura 1.11.0 installed gives
functions with 1.11.0's exact signatures.

## The datatrove step

`pyroid.integrations.datatrove.Trafilatura` replaces datatrove 0.10.1's `Trafilatura` extractor
(`pip install 'pyroid[datatrove]'`):

```diff
-from datatrove.pipeline.extractors import Trafilatura
+from pyroid.integrations.datatrove import Trafilatura

 pipeline = [
     WarcReader(...),
     URLFilter(),
     Trafilatura(favour_precision=True),
     ...
 ]
```

It takes datatrove's arguments with their defaults (`favour_precision=True`,
`include_images=False`, `timeout=1`, `deduplicate=True`, and `**kwargs` passed to `extract`),
plus three keyword-only ones:

| Argument | Default | Meaning |
|---|---|---|
| `compat` | `None` | the trafilatura release to reproduce. `None`: the installed trafilatura's version if pyroid reproduces it (2.3.1 or 1.11.0); 2.3.1 if trafilatura is not installed; for any other installed version, datatrove's own step runs the documents (with one `FallbackWarning`) |
| `batch_size` | `128` | documents per pyroid call |
| `threads` | `1` | pyroid's `threads=` ([Threads](../concepts/threads.md)) |

A stock `datatrove[processing]` install has trafilatura 1.11.0, so with `compat=None` the step
runs pyroid's 1.11.0 mode, and a FineWeb pipeline gives the same documents as with datatrove's own
step. With trafilatura 2.3.1 installed it runs the 2.3.1 mode. trafilatura is not needed at all;
without it the step runs 2.3.1 unless `compat="1.11.0"` says otherwise.

The documents out, their text, and the statistics (`total`, `extracted`, `forwarded`, `dropped`,
`doc_len`) are datatrove's. The step runs in the task's process, so it differs from datatrove's
in these ways:

- There is no child process and no timeout. A page datatrove gives up on after `timeout`
  seconds (counted as `timeout`, and followed by a fresh child with an empty dedup cache) gets its
  text here.
- With `deduplicate=True` the dedup cache is a fresh `LRUCache` per `run()`, that is per task,
  as datatrove's per-task child process has a fresh `LRU_TEST`.
- Errors are counted as datatrove counts them. A page whose extraction raises kills datatrove's
  child, so it is `broken_process`, and the rest of the task continues with an empty dedup cache,
  as datatrove's next child does. Arguments that make `extract("")` raise (datatrove's warm-up
  call, for example a keyword `extract` does not take) make every page `timeout`.
- Time is tracked per batch, not per document.

## `pyroid.patch("trafilatura")`

```python
import pyroid

pyroid.patch("trafilatura")
from trafilatura import extract  # pyroid's, with trafilatura's exact signature
```

The patch puts shims in `sys.modules` for `trafilatura` and `trafilatura.core` (`extract`,
`bare_extraction`, and with 2.3.1 `extract_with_metadata`), `trafilatura.deduplication`
(`LRUCache`, `LRU_TEST`) and `trafilatura.meta` (`reset_caches`, which empties pyroid's cache
too). Everything else comes from the installed trafilatura. The mode follows the installed
release: with 1.11.0 installed the shim's functions are 1.11.0's (its `extract` takes
`**kwargs` and has no `fast`, its `bare_extraction` returns a dict by default), with 2.3.1 they
are 2.3.1's. Either way they have exactly the installed trafilatura's signatures, without
`compat`, and `config` defaults to its `DEFAULT_CONFIG`. With another release installed, a
`PatchWarning` says that the patched functions give trafilatura 2.3.1's results. datatrove's own `Trafilatura` step
does `from trafilatura import extract` in its child process, so a patch made before the pipeline
starts (or `pyroid run -p trafilatura`) reaches it. See [`pyroid.patch()`](patch.md).

## Differences from trafilatura

From the module's documentation (`pyroid/trafilatura.pyi`):

- Trafilatura's log messages (`logging.getLogger("trafilatura")`) are not emitted.
- `config` defaults to `pyroid.trafilatura.DEFAULT_CONFIG`, not
  `trafilatura.settings.DEFAULT_CONFIG`: changing trafilatura's object changes nothing here
  (under `pyroid.patch` the shim's functions do default to trafilatura's object).
- `LRUCache` keys are str without lone surrogates, and its `cache` is a read-only copy.
- `bare_extraction`'s `body` and `commentsbody` become lxml elements when first read, and are the
  XML strings without lxml.

And by design, the [delegated options](#delegated-options) need the trafilatura release of the
mode installed.

## Parity

On the 4,677 Common Crawl pages of the development sample (CC-MAIN-2026-39, as datatrove's
`WarcReader` and `URLFilter` give them), every page gives trafilatura 2.3.1's text in 4 option
sets (datatrove's call; without dedup; `fast`; `fast` with dedup), both page by page and through
`extract_batch`: 0 differences in 37,416 comparisons, with dedup runs in page order. The test
suite (`python/tests/test_trafilatura*.py`) also compares pyroid with trafilatura on pages from a
Hypothesis tag grammar and on Common Crawl pages cut, spliced and with junk inserted, in 7 option
sets, `bare_extraction` field by field, the dedup cache's contents after each sequence, every
delegated option (one `FallbackWarning`, trafilatura's result), and the datatrove step against
datatrove's own, statistics included. The parity suite (`parity/`, run on 2026-11-09 on CPython
3.13) found 0 differences on its 3 pinned corpora of 7,192 Common Crawl pages in 7 option sets
and 2 `bare_extraction` sets, and on Trafilatura's own test suite replayed as 2,669 native and
1,884 delegated calls; details on the [Compatibility](../compatibility.md) page.

For `compat="1.11.0"`, the same 4,677 pages give trafilatura 1.11.0's text in 5 option sets
(datatrove's call, without dedup, `no_fallback`, the defaults, the defaults with dedup; 7 on the
1,000 development pages), page by page and through `extract_batch`: 0 differences in 50,770
comparisons. The test suite runs the same checks against trafilatura 1.11.0 (with lxml 6.1.3 and
jusText 3.0.2) when it is installed, and the datatrove step against datatrove's own step on
trafilatura 1.11.0, as a stock `datatrove[processing]` install has it.

## Speed

Measured on 1,000 Common Crawl pages (157,000 characters each on average) on a Docker/OrbStack
VM on Apple Silicon (12 vCPUs, load 2 to 4), CPython 3.13, indicative. Each mode against its own
trafilatura release:

| Call | `compat="2.3.1"` vs trafilatura 2.3.1 | `compat="1.11.0"` vs trafilatura 1.11.0 |
|---|---:|---:|
| `extract(page)`, one page at a time | 4.4 vs 31.6 ms/page (7.2×) | 2.7 vs 23.4 ms/page (8.6×) |
| `extract_batch`, 1 / 4 / 8 threads | 4.1 / 0.99 / 0.50 ms/page | 2.7 / 0.68 / 0.33 ms/page |
| datatrove's call (`favor_precision`, `deduplicate=True`), one at a time | 3.5 vs 27.3 ms/page (7.8×) | 3.0 vs 32.4 ms/page (10.8×) |
| the same with `extract_batch`, 1 / 4 / 8 threads | 3.4 / 2.0 / 2.0 ms/page | 2.9 / 2.1 / 2.1 ms/page |
| datatrove's `Trafilatura` step, one task, vs datatrove's own step | 3.0 (`threads=1`), 2.1 (`threads=8`) vs 26.3 ms/page (8.7×) | 2.9 (`threads=1`), 2.2 (`threads=8`) vs 32.9 ms/page (11×) |

With deduplication, extraction runs in page order, so threads only parse ahead and the gain
stops at about 1.5×. A datatrove pipeline gets its parallelism from running one task per core.

### datatrove's FineWeb pipeline from WARC files

`bench/fineweb_warc.py` runs datatrove 0.10.1's `examples/fineweb.py` (WARC reader, URL filter,
Trafilatura, language ID, the quality filters, MinHash dedup, PII) on 8 WARC files of
CC-MAIN-2026-39 (171,595 HTML records), as stock datatrove with trafilatura 1.11.0, with
pyroid's drop-in steps in datatrove's executor, and as a native pyroid chain. AWS EC2
c7i.4xlarge (Intel Xeon Platinum 8488C), CPython 3.13.18, 8 workers on 8 cores or 1 worker:

| Run | 8 workers | 1 worker |
|---|---:|---:|
| stock datatrove | 871 s | 6,976 s |
| datatrove with pyroid's steps | 104 s (8.4×) | 776 s (9.0×) |
| native pyroid chain | 110 s | 739 s (9.4×) |

The Trafilatura step took 5,543 s of stock datatrove's 6,759 CPU seconds at 8 workers, and 514 of
792 with pyroid's steps. Of the steps still in Python, the largest is datatrove's WARC reader
(warcio): 154 of those 792 CPU seconds.

Stock datatrove gives up on a page after a 1 s timeout, so which pages it drops varies between
runs: 69 at 8 workers and 80 at 1 worker, and the two runs kept 23,503 and 23,501 documents.
pyroid has no timeout and kept 23,505 in every run, with the same final documents from its steps
and from the native chain. With the pages stock datatrove dropped left out the same way, the
native chain's final documents are byte-identical to stock datatrove's, for both stock runs. Both
also drop the 5 pages on which trafilatura 1.11.0 itself raises. The results are in
`bench/results/aws-c7i.4xlarge/2026-11-09-warc/`.
