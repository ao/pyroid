# pyroid

pyroid reimplements 21 popular Python libraries in Rust, with the same API and the same output.
Most of its modules are many times faster than the library they replace. They cover data formats
(JSON, YAML, JMESPath queries, dates, tables), developer tools (difflib, packaging, sqlparse), text
and NLP (ftfy, fastText, langdetect, NLTK, jieba), web pages and archives, and the steps of LLM data
pipelines. It runs on CPython 3.11 to 3.15, including free-threaded 3.14t and 3.15t.

```console
$ pip install pyroid
```

Change the import. The output is the same:

```python
import pyroid.yaml as yaml  # was: import yaml
import pyroid.jmespath as jmespath  # was: import jmespath
import pyroid.ftfy as ftfy  # was: import ftfy
from pyroid.packaging.version import Version  # was: from packaging.version import Version

yaml.safe_load("ports: [80, 443]")  # {'ports': [80, 443]}
jmespath.search("items[?up].name", {"items": [{"name": "web", "up": True}]})  # ['web']
ftfy.fix_text("naÃ¯ve rÃ©sumÃ©")  # 'naïve résumé'
Version("1.10") > Version("1.9")  # True
```

For code you don't own, such as boto3 or a pipeline library, patch its imports instead:

```python
import pyroid

pyroid.patch("jmespath", "ftfy")  # from here on, jmespath.search and ftfy.fix_text are pyroid's

import jmespath

jmespath.search("a.b", {"a": {"b": 1}})  # 1, computed by pyroid
```

See [`pyroid.patch()`](guides/patch.md) for the libraries it covers.

## What's in it

| Area | Module | Replaces | Identical output? |
|---|---|---|---|
| Data formats | [`pyroid.json_repair`](guides/json_repair.md) | [json_repair](https://github.com/mangiucugna/json_repair) 0.64.9 | same output, linear where json_repair is still quadratic ([parity](compatibility.md#json_repair)) |
|  | [`pyroid.yaml`](guides/yaml.md) | [PyYAML](https://github.com/yaml/pyyaml) 6.0.3's `safe_load` and `safe_dump` | same objects, text and errors as PyYAML's pure-Python `SafeLoader` and `SafeDumper`, faster than `CSafeLoader` ([parity](compatibility.md#yaml)) |
|  | [`pyroid.jmespath`](guides/jmespath.md) | [jmespath](https://github.com/jmespath/jmespath.py) 1.1.0 | same results, ASTs and exceptions as jmespath, on your Python objects ([parity](compatibility.md#jmespath)) |
|  | [`pyroid.tabulate`](guides/tabulate.md) | [tabulate](https://github.com/astanin/python-tabulate) 0.10.0, `DataFrame.to_markdown()` included | the same text, byte for byte, in every format ([parity](compatibility.md#tabulate)) |
|  | [`pyroid.dateutil.parser`](guides/dateutil.md) | [python-dateutil](https://github.com/dateutil/dateutil)'s `parse` and `isoparse`, 2.8.2 to 4.0.0 | same datetimes, tz objects, exceptions and messages ([parity](compatibility.md#dateutil)) |
| Developer tools | [`pyroid.difflib`](guides/difflib.md) | the standard library's [`difflib`](https://docs.python.org/3/library/difflib.html) (CPython 3.11 to 3.15) | the interpreter's own ratios (float bits), opcodes, diff lines and HTML ([parity](compatibility.md#difflib)) |
|  | [`pyroid.packaging`](guides/packaging.md) | [packaging](https://github.com/pypa/packaging) 26.3's versions, specifiers, markers, requirements and name helpers | same parsing, order, hashes, strings and exceptions; packaging's own tests pass under the patch ([parity](compatibility.md#packaging)) |
|  | [`pyroid.sqlparse`](guides/sqlparse.md) | [sqlparse](https://github.com/andialbrecht/sqlparse) 0.6.1 | same formatted text, statements, token trees (sqlparse's own classes) and exceptions ([parity](compatibility.md#sqlparse)) |
| Text and encoding | [`pyroid.ftfy`](guides/ftfy.md) | [ftfy](https://github.com/rspeer/python-ftfy) 6.3.1 | byte-identical ([parity](compatibility.md#ftfy)) |
|  | [`pyroid.translit`](guides/translit.md) | `anyascii` 0.3.3; a permissive alternative to GPL `Unidecode` | identical to `anyascii`; not Unidecode's output |
| Language ID and NLP | [`pyroid.fasttext`](guides/fasttext.md) | [fastText](https://github.com/facebookresearch/fastText) inference (`fasttext`, `fasttext-predict`, ...) | bit-identical labels and probabilities on Linux (glibc) and Windows x64; on macOS the probabilities can differ in the last bits ([parity](compatibility.md#fasttext)) |
|  | [`pyroid.nltk`](guides/nltk.md) | [NLTK](https://www.nltk.org) 3.10.4's `sent_tokenize` (Punkt), `word_tokenize`, Treebank and regexp tokenizers | same sentences, tokens and offsets ([parity](compatibility.md#nltk)) |
|  | [`pyroid.jieba`](guides/jieba.md) | [jieba](https://github.com/fxsjy/jieba) 0.42.1 | same words, offsets and frequencies ([parity](compatibility.md#jieba)) |
|  | [`pyroid.langdetect`](guides/langdetect.md) | [langdetect](https://github.com/Mimino666/langdetect) 1.0.9 | same languages and probabilities, bit for bit, for the same seed ([parity](compatibility.md#langdetect)) |
| Web and archives | [`pyroid.trafilatura`](guides/trafilatura.md) | [Trafilatura](https://github.com/adbar/trafilatura)'s text extraction, 2.3.1 or 1.11.0 (the release datatrove 0.10.1 installs), with lxml 6.1.3's HTML parser and jusText 3.0.2 | the same text, byte for byte ([parity](compatibility.md#trafilatura)) |
|  | [`pyroid.warc`](guides/warc.md) | [warcio](https://github.com/webrecorder/warcio) 1.8.2's WARC/ARC reader, and datatrove 0.10.1's WARC reading | same records, documents, exceptions and messages ([parity](compatibility.md#warc)) |
| LLM data: filters and dedup | [`pyroid.filters`](guides/filters.md) | [datatrove](https://github.com/huggingface/datatrove) 0.10.1's Gopher, C4 and FineWeb filters, PII formatter, C4 bad words, symbol lines and URL filter | same decisions, reasons and text ([parity](compatibility.md#filters)) |
|  | [`pyroid.dedup`](guides/dedup.md) | datatrove 0.10.1's MinHash and sentence dedup | same signatures, removals and files ([parity](compatibility.md#dedup)) |
|  | [`pyroid.datasketch`](guides/datasketch.md) | [datasketch](https://github.com/ekzhu/datasketch) 2.0.0's MinHash, LeanMinHash, MinHashLSH, LSH Forest and LSH Ensemble | same hash values, bytes and query results ([parity](compatibility.md#datasketch)) |
|  | [`pyroid.nemo`](guides/nemo-curator.md#pyroidnemo) | [NeMo Curator](https://github.com/NVIDIA-NeMo/Curator) 1.4.0's heuristic document filters and string modifiers | same scores, exceptions and modified text ([parity](compatibility.md#nemo)) |
| Evaluation | [`pyroid.metrics`](guides/metrics.md) | [rouge-score](https://github.com/google-research/google-research/tree/master/rouge) 0.1.2 and [sacrebleu](https://github.com/mjpost/sacrebleu) 2.6.0 (BLEU, chrF, chrF++) | identical scores and signatures, no tolerance ([parity](compatibility.md#metrics)) |

## Example: cleaning LLM training data

Building a pretraining or fine-tuning corpus means running every document through the same few
steps: fix broken Unicode, identify the language, drop low-quality pages, remove near-duplicates.
The usual tools for them are pure Python (ftfy, datatrove's filters and MinHash) or C++ extensions
that lag behind new Pythons (fastText).

On one core, [datatrove](https://github.com/huggingface/datatrove)'s FineWeb chain spends about
79% of its CPU in the four quality filters and about 18% in MinHash. ftfy is 2.3% and language ID
1% (the first baseline run, in a Docker VM: [method and caveats](benchmarks.md#where-the-cpu-goes)).
The filters and MinHash both tokenize every document with spaCy and then do their own work in
Python. pyroid moves both to Rust.

From raw WARC files, text extraction with Trafilatura comes first, and it costs more than all the
other steps together. On 8 Common Crawl WARC files, stock datatrove spent 5,403 of its 6,593 CPU
seconds in it (AWS EC2 c7i.4xlarge, 8 workers; the run behind the 10.7× below). [`pyroid.trafilatura`](guides/trafilatura.md)
ports that step too.

On 100,000 FineWeb documents, pyroid's chain is 27× faster than datatrove on 8 processes, with
the same output. From WARC files, `pyroid clean` runs the whole pipeline 10.7× faster (AWS EC2
c7i.4xlarge; see [Speed](#speed)).

The command line runs that chain (ftfy, then FineWeb's language ID, filters and MinHash) over
Parquet or JSONL, with the same output as datatrove's stages:

```console
$ pip install "pyroid[cli]"
$ pyroid clean in.parquet out.parquet --recipe fineweb
```

See [the command line](guides/cli.md). In Python, the same steps are a few calls:

```python
import pyroid.ftfy as ftfy  # was: import ftfy
import pyroid.fasttext as fasttext  # was: import fasttext
import pyroid.filters as filters  # datatrove's Gopher / C4 / FineWeb filters

docs = ["naÃ¯ve rÃ©sumÃ©", "Some other web page ..."]

# ["naïve résumé", ...], like [ftfy.fix_text(d) for d in docs] but on all cores
docs = ftfy.fix_text_batch(docs)

# fastText's own .bin / .ftz files; a batch runs without the GIL. lid.176.bin is fastText's
# language-ID model (125 MiB): https://fasttext.cc/docs/en/language-identification.html
model = fasttext.load_model("lid.176.bin")
labels, probs = model.predict(docs, k=1)  # [["__label__fr"], ["__label__en"]]

# datatrove's keep/drop decision per document
keep = filters.gopher_quality(docs)  # [False, False]: both are under 50 words
```

Batch functions take a `str`, a `list` or `tuple` of `str`, or an Arrow string column (pyarrow,
polars, pandas, Hugging Face `datasets`), and return text and masks in the same kind
([Inputs and outputs](concepts/inputs.md)).

The filters and dedup tokenize 4,049 of datatrove's 4,343 language codes natively (Japanese with
an installed SudachiDict dictionary, Thai with an installed PyThaiNLP) and send the rest through
datatrove's own tokenizers ([Languages](guides/languages.md)).

## Install

```console
$ pip install pyroid               # no dependencies
$ pip install "pyroid[fasttext]"   # adds numpy, which pyroid.fasttext returns arrays in
$ pip install "pyroid[cli]"        # the pyroid command line (pyarrow)
```

`pyroid.fasttext`, `pyroid.datasketch` and the batch calls of `pyroid.nemo` need numpy, which
`[fasttext]`, `[datasketch]` and `[nemo]` add (`import pyroid.nemo` works without it).
`[datatrove]` and `[hf]` add the datatrove steps and the Hugging Face helpers. There are prebuilt
wheels for Linux (glibc and musl), macOS and Windows, on x86-64 and ARM64. See
[Install](install.md).

## Integrations

| Integration | What it does |
|---|---|
| [`pyroid clean`](guides/cli.md) | the FineWeb-style chain as one command over Parquet or JSONL, or datatrove's FineWeb pipeline from WARC files, with datatrove's output document for document |
| [Large datasets](guides/large-datasets.md) | the same commands over thousands of files: worker processes, `--shard I/N` across machines, `s3://`, `gs://` and `hf://` inputs, resumable runs, and MinHash dedup across shards in datatrove's file format |
| [`pyroid.patch()`](guides/patch.md) | makes `import ftfy`, `fasttext`, `anyascii`, `datasketch`, `json_repair`, `trafilatura`, `nltk`, `warcio`, `nemo_curator`, `yaml`, `tabulate`, `dateutil`, `jmespath`, `packaging`, `sqlparse` and `langdetect` give pyroid's functions, in code you don't own, and patches `difflib` in place |
| [datatrove steps](guides/datatrove.md) | `pyroid.integrations.datatrove`: datatrove's pipeline steps with the same names and signatures, `Trafilatura` included, as a one-import change |
| [NeMo Curator stages](guides/nemo-curator.md) | `pyroid.integrations.nemo_curator`: NeMo Curator 1.4.0's CPU text stages, subclassing NeMo's classes with the same arguments |
| [Hugging Face `datasets`](guides/huggingface.md) | `pyroid.integrations.hf`: batched callables for `Dataset.map` and `Dataset.filter`, and the FineWeb recipe |

## Speed

From Common Crawl WARC files to deduplicated documents, stock datatrove's FineWeb pipeline took
848.3 s with 8 workers. With pyroid's steps in datatrove's executor, WARC reader included, it took
89.4 s (9.5×), and `pyroid clean --recipe fineweb-warc` took 79.6 s (10.7×). The input was 8 WARC
files of CC-MAIN-2026-39 (171,595 HTML records), on an AWS EC2 c7i.4xlarge (Intel Xeon Platinum
8488C) pinned to 8 CPUs. The kept documents are the same once the pages stock datatrove drops at
its 1 s extraction timeout are left out of pyroid's run too
([From WARC files](benchmarks.md#from-warc-files-datatroves-fineweb-pipeline)).

On 100,000 FineWeb documents, with the text already extracted, pyroid's chain took 10.2 s on 8
threads against 272.1 s for datatrove on 8 processes (about 27×), and 43.4 s against 2,444.4 s on
one (about 56×), with identical output, on the same instance type. On the 10,000-document dev
slice the ratios are 18× and 38×. On a c8g.4xlarge (AWS Graviton4) the 100k run gives 29× and
50×.

Per stage, on the 10k dev slice and the c7i, pyroid on one thread against the incumbent in one
process:

| Module | Incumbent | Speedup |
|---|---|---:|
| `pyroid.ftfy` | ftfy 6.3.1 | 108× |
| `pyroid.fasttext` | fasttext-predict 0.9.2.4 | 1.6× (`.bin`), 3.6× (`.ftz`) |
| `pyroid.filters` | datatrove 0.10.1's Gopher, C4 and FineWeb filters | 35–69× |
| `pyroid.dedup` | datatrove 0.10.1's MinHash signatures | 34× |
| `pyroid.translit` | anyascii 0.3.3 | 53× |

The other modules have only been measured in a Docker/OrbStack VM on Apple Silicon or on a shared
EC2 machine, so treat their numbers as indicative (one thread unless stated):

| Module | Incumbent | Speedup |
|---|---|---:|
| `pyroid.trafilatura.extract`, default options (`compat="2.3.1"` / `"1.11.0"`) | Trafilatura 2.3.1 / 1.11.0 | 7.2× / 8.6× |
| `pyroid.metrics`, ROUGE | rouge-score 0.1.2 | 59–65× |
| `pyroid.metrics`, corpus BLEU / chrF | sacrebleu 2.6.0 | 11× / 19× |
| `pyroid.json_repair.repair_json`, broken JSON | json_repair 0.64.9 | about 15× |
| `pyroid.datasketch`, MinHash signatures, 1 / 8 threads | datasketch 2.0.0 | 8× / 21× |
| `pyroid.jieba.lcut`, a batch, 1 / 8 threads | jieba 0.42.1 | 24× / 58× |
| `pyroid.nltk.sent_tokenize` / `word_tokenize` | NLTK 3.10.4 | 60× / 31× |
| `pyroid.filters`, FineWeb-2 settings, Chinese / Japanese | datatrove 0.10.1 | 44× / 7.8× |

The [Benchmarks](benchmarks.md) page has every number, how it was measured and what it doesn't
show.

## Correctness

Each module is tested differentially against the library it replaces. The inputs are pinned
FineWeb and FineWeb-2 documents, pinned Common Crawl pages, pinned evaluation data, the
incumbent's own test corpus, and Hypothesis fuzzing that includes lone surrogates, mojibake and
every option. The counts are on the [Compatibility](compatibility.md) page, and every known
difference is listed in each guide.

All suites ran on Linux aarch64 in a Docker VM, and CI runs the `parity/` suites on Linux x86_64
and aarch64. The ftfy, fastText, filters, dedup, metrics and json_repair suites also ran natively on
x86_64 and aarch64 on AWS EC2, and the Trafilatura pages on x86_64, with 0 differences
([Platforms](compatibility.md#platforms)). On macOS, Windows and musl Linux, every release's
wheels pass pyroid's test suite before they are published.

Where a module meets an option or input it does not port, it hands the call to the real library
with a [`FallbackWarning`](concepts/fallbacks.md), or raises an error.

Results don't depend on the number of threads or workers, or on the machine's load. Stock
datatrove's Trafilatura step does: it drops whichever pages take more than 1 s.

## Licence

pyroid's own code is MIT. The ported parts keep their licences, so the package's licence
expression is `MIT AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause AND ISC AND
PSF-2.0 AND Unicode-3.0 AND Zlib`:

- Apache-2.0 for the ftfy, datatrove, rouge-score, NLTK (Porter stemmer and tokenizers),
  sacrebleu, Trafilatura, SudachiPy, warcio and langdetect ports (and langdetect's profiles);
- MIT for the fastText, spaCy, indic-nlp-library, urduhack, json_repair, datasketch, jieba (and
  its dictionary), libxml2 (HTML parser), PyYAML, tabulate and jmespath ports;
- BSD-2-Clause for the jusText port and its stoplists, and the packaging port (packaging is
  dual-licensed Apache-2.0 OR BSD-2-Clause);
- BSD-3-Clause for numpy's random generator, scipy's QUADPACK integration, tldextract, lxml,
  python-dateutil (whose later contributions are also Apache-2.0), sqlparse and the Mersenne
  Twister (MT19937);
- PSF-2.0 for CPython's `ipaddress`, float formatting, `difflib` and `random`;
- ISC for anyascii's data;
- Unicode-3.0 for Unicode Character Database data;
- Zlib for zlib-rs, the Rust crate behind `pyroid.warc`'s gzip decoding.

The [NOTICE](https://github.com/ao/pyroid/blob/main/NOTICE) file lists the ported files, and the
licence texts are in [LICENSES/](https://github.com/ao/pyroid/tree/main/LICENSES), with
those of the Rust crates the extension links in `THIRD-PARTY-RUST.txt` there.
