# `pyroid.nltk`: drop-in NLTK tokenizers

`pyroid.nltk` is a Rust port of the sentence and word tokenizers of [NLTK](https://www.nltk.org)
3.10.4 (Apache-2.0): Punkt (`sent_tokenize`, `PunktSentenceTokenizer`, `PunktTokenizer`),
`word_tokenize`, the Treebank tokenizers and detokenizer, and the common
[regular-expression tokenizers](#regular-expression-tokenizers). It returns the same sentences,
tokens and offsets as NLTK 3.10.4 for every input ([parity](../compatibility.md#nltk)), several
times faster on one thread, and batches run in parallel.

## Change one import

```diff
-import nltk
+import pyroid.nltk as nltk

 nltk.sent_tokenize("Mr. Smith went to Washington. He liked it.")
 # ['Mr. Smith went to Washington.', 'He liked it.']
 nltk.word_tokenize("They'll pay $3.88 (roughly).")
 # ['They', "'ll", 'pay', '$', '3.88', '(', 'roughly', ')', '.']
```

Or keep `import nltk` and call [`pyroid.patch("nltk")`](patch.md) once: NLTK's own
`nltk.sent_tokenize`, `nltk.word_tokenize` and the tokenizer classes are then pyroid's, and
`pyroid.unpatch("nltk")` puts them back.

## The Punkt models

Punkt needs NLTK's pretrained `punkt_tab` models. pyroid does not bundle them, because
nltk_data publishes them without a licence. Install them as for NLTK:

```bash
python -m nltk.downloader punkt_tab     # or, in Python: nltk.download("punkt_tab")
```

With NLTK installed, pyroid finds and reads the model with NLTK's own `nltk.data.find` and
`load_punkt_params`, so `nltk.data.path`, `NLTK_DATA` and NLTK's checks apply as usual. NLTK is
not needed otherwise. Without it, pyroid looks where NLTK looks by default (`NLTK_DATA`,
`~/nltk_data`, and the `nltk_data` directories under `sys.prefix`, `/usr/share`,
`/usr/local/share`, `/usr/lib` and `/usr/local/lib`) for an unpacked
`tokenizers/punkt_tab/<language>/` or for `tokenizers/punkt_tab.zip`. A missing model raises
NLTK's `LookupError`.

All 19 models of `punkt_tab` work: czech, danish, dutch, english, estonian, finnish, french,
german, greek, italian, malayalam, norwegian, polish, portuguese, russian, slovene, spanish,
swedish and turkish.

## Sentences

```python
import pyroid.nltk as nltk

nltk.sent_tokenize("Dr. Müller kam um 9 Uhr. Er war müde.", language="german")
# ['Dr. Müller kam um 9 Uhr.', 'Er war müde.']

tok = nltk.PunktTokenizer("english")
list(tok.span_tokenize("One. Two."))  # [(0, 4), (5, 9)]
tok.tokenize('"Go." He went.')  # ['"Go."', 'He went.']: the quote stays with its sentence
tok.tokenize('"Go." He went.', realign_boundaries=False)  # ['"Go.', '" He went.']
```

`PunktSentenceTokenizer` takes a `PunktParameters`, as in NLTK. Its sets can be changed in place,
and the tokenizer sees the change at its next call:

```python
tok = nltk.PunktTokenizer()
tok.tokenize("Sales rose approx. 4%. The dept. agreed.")
# ['Sales rose approx.', '4%.', 'The dept.', 'agreed.']
tok._params.abbrev_types.update({"approx", "dept"})
tok.tokenize("Sales rose approx. 4%. The dept. agreed.")
# ['Sales rose approx. 4%.', 'The dept. agreed.']
```

`span_tokenize`, `sentences_from_text`, `text_contains_sentbreak`, `sentences_from_tokens`,
`sentences_from_text_legacy` and `debug_decisions` are there too. A `PunktLanguageVars` subclass
that changes `sent_end_chars` (for example to add `"…"`) still runs in Rust.

ROUGE in [`pyroid.metrics`](metrics.md) uses this port for `split_summaries=True`, so rougeLsum's
sentence split runs in Rust too.

## Words

`word_tokenize` splits the Punkt sentences with `NLTKWordTokenizer`, NLTK's improved Treebank
rules. With `preserve_line=True` it skips the sentence split, and needs no model.

```python
import pyroid.nltk as nltk

nltk.word_tokenize("I can't go to «Paris» -- it's 3:30.")
# ['I', 'ca', "n't", 'go', 'to', '«', 'Paris', '»', '--', 'it', "'s", '3:30', '.']
nltk.TreebankWordTokenizer().tokenize("Good muffins cost $3.88\nin New York.")
# ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York', '.']
list(nltk.NLTKWordTokenizer().span_tokenize("Hello, world."))  # [(0, 5), (5, 6), (7, 12), (12, 13)]
nltk.TreebankWordDetokenizer().detokenize(["I", "ca", "n't", "go", "."])  # "I can't go."
```

`convert_parentheses=True` gives the Penn Treebank symbols (`-LRB-`, `-RRB-`, ...), as in NLTK.
`align_tokens(tokens, sentence)` gives the offsets of tokens found one after another.

## Regular-expression tokenizers

`RegexpTokenizer`, `WhitespaceTokenizer`, `BlanklineTokenizer`, `WordPunctTokenizer`,
`regexp_tokenize`, `wordpunct_tokenize` and `blankline_tokenize` are NLTK's, with `gaps`,
`discard_empty` and `span_tokenize`:

```python
import pyroid.nltk as nltk

nltk.wordpunct_tokenize("Good muffins cost $3.88\nin New York.")
# ['Good', 'muffins', 'cost', '$', '3', '.', '88', 'in', 'New', 'York', '.']
nltk.RegexpTokenizer(r"\w+|\$[\d\.]+|\S+").tokenize("Good muffins cost $3.88\nin New York.")
# ['Good', 'muffins', 'cost', '$3.88', 'in', 'New', 'York', '.']
```

These patterns run in Rust when spelled exactly so, with no flags beyond NLTK's default
`re.UNICODE | re.MULTILINE | re.DOTALL`: `\w+`, `\w+|[^\w\s]+`, `\w+|\$[\d\.]+|\S+`, `[\w']+`,
`\S+`, `\s+`, `[a-zA-Z]+` and `BlanklineTokenizer`'s. Any other pattern runs in NLTK.

## Many documents

A list or tuple of `str`, or an Arrow string column, gives one list per document, computed in
parallel on pyroid's shared pool. Arrow nulls give `None`.

```python
import pyroid.nltk as nltk

docs = ["One. Two.", "Three."]
nltk.sent_tokenize(docs)  # [['One.', 'Two.'], ['Three.']]
nltk.word_tokenize(docs, threads=8)  # at most 8 tasks; threads=1 runs on the calling thread
nltk.PunktTokenizer("french").span_tokenize(docs)  # [[(0, 4), (5, 9)], [(0, 6)]]
```

`threads=` is keyword-only and follows the [threading conventions](../concepts/threads.md). On
free-threaded CPython the GIL stays off.

## What NLTK still does

These are handed to NLTK 3.10.4, which must then be installed. The first call emits a
[`pyroid.FallbackWarning`](../concepts/fallbacks.md); without NLTK it raises
[`pyroid.PyroidError`](../reference/pyroid.md#pyroid.PyroidError).

- Training: `PunktTrainer`, and `PunktSentenceTokenizer(train_text)` with a str. The trained
  parameters then tokenize in Rust.
- A `PunktSentenceTokenizer` with a custom `token_cls`, or with a `PunktLanguageVars` that
  overrides more than `sent_end_chars`; parameters that are not sets of str.
- `dump`, `PunktTokenizer.save_params` and `save_dir`.
- A Treebank tokenizer or detokenizer whose pattern lists were edited or overridden.
- A `RegexpTokenizer` pattern that is not one of the native ones.

The other NLTK tokenizers (`TweetTokenizer`, `ToktokTokenizer`, ...) are not in `pyroid.nltk`;
keep using NLTK's.

## Differences from NLTK

- A `str` with a lone surrogate (`"\ud800"`) raises `UnicodeEncodeError`, as everywhere in
  pyroid.
- NLTK runs every pattern under a 5-second timeout (`nltk.redos`) and raises `TimeoutError`
  when a match takes longer; pyroid has no such limit.
- NLTK's patterns run on the `regex` module. pyroid reproduces `regex` 2026.9.29's character
  classes. With another `regex` release, NLTK can differ on characters whose Unicode properties
  changed between the two releases.
- Tokenizing does not add a 0 entry to `PunktParameters.ortho_context` for each unseen word, as
  NLTK's `defaultdict` does.
- An NLTK `PunktParameters` given to a tokenizer is copied, so edit it through `tok._params`;
  later edits of NLTK's object are not seen.
- NLTK compiles a `PunktLanguageVars` object's patterns at first use and keeps the old
  `sent_end_chars` in part of the algorithm after they change; pyroid uses the new ones
  throughout.

## Parity

Every check found 0 differences from NLTK 3.10.4:

| Check | Comparisons |
|---|--:|
| Parity suite against live NLTK, every document: Punkt in all 19 models on FineWeb, FineWeb-2 and the adversarial set, `word_tokenize`, both Treebank tokenizers and their spans, detokenize round trips, the regexp tokenizers, the batch paths | 14,174,241 |
| All 19 `punkt_tab` models on 39,618 FineWeb and FineWeb-2 documents (848,037 sentences) | 178,282 |
| The Treebank tokenizers on 600,372 sentences and documents of FineWeb and 27 FineWeb-2 languages | 5,713,962 |
| Every code point, in templates around the Treebank, detokenizer and regexp rules | 149,016,576 |
| Recorded NLTK outputs replayed in the Rust tests: NLTK's own examples, adversarial text, fuzz, and the case rules of CPython 3.12 to 3.15 | 77,535 |

Hypothesis properties also compare every function against live NLTK, on CPython 3.13 and 3.15t,
as do NLTK's doctest examples and every code point alone. See
[Compatibility](../compatibility.md#nltk).

## Speed

Docker/OrbStack VM on Apple Silicon (12 vCPU, load 3 to 5), indicative. These are the first
1,000 documents of the FineWeb dev slice (3,018 characters each on average), with release
wheels and the English model. Medians of 5 runs (NLTK: 2), with every document's output checked
against NLTK's first. µs per document:

| Run | `sent_tokenize`, 3.13 | `word_tokenize`, 3.13 | `sent_tokenize`, 3.15t | `word_tokenize`, 3.15t |
|---|--:|--:|--:|--:|
| NLTK, one thread | 651 | 3,576 | 592 | 3,513 |
| NLTK, 8 threads | 4,102 | 25,633 | 259 | 1,244 |
| NLTK, 8 processes | 197 | 593 | 117 | 724 |
| pyroid, one call per document | 10.9 (60×) | 117 (31×) | 11.2 (53×) | 105 (33×) |
| pyroid, batch, `threads=1` | 10.4 (63×) | 92 (39×) | 10.5 (56×) | 96 (37×) |
| pyroid, batch, `threads=8` | 5.6 (116×) | 44 (82×) | 5.0 (118×) | 43 (81×) |

On 8 threads the batch gains about 2× over one thread: building the Python lists of sentences
and tokens runs on the calling thread. Against NLTK on 8 processes, pyroid on 8 threads is 35×
faster for `sent_tokenize` and 14× for `word_tokenize` on CPython 3.13. The first sentence in a
fresh interpreter takes about as long as NLTK's (177 ms against 203 ms on 3.13), because pyroid
imports NLTK to find the model when NLTK is installed. Source:
`bench/results/orbstack-aarch64-12cpu/2026-11-09-nltk/`.

API: [`pyroid.nltk` reference](../reference/nltk.md).
