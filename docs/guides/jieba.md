# `pyroid.jieba`: drop-in jieba

`pyroid.jieba` is a Rust port of [jieba](https://github.com/fxsjy/jieba) 0.42.1 by Sun Junyi
(MIT), the Chinese word segmenter. It returns the same words, offsets, frequencies and exceptions
as jieba 0.42.1 for every input ([parity](../compatibility.md#jieba)). jieba's default dictionary
is built in, so loading it takes about 3 ms instead of jieba's 0.4 s. Segmentation runs about 20×
faster per document on one thread, and batches run in parallel. The same port tokenizes Chinese in
[`pyroid.filters` and `pyroid.dedup`](languages.md).

## Change one import

```diff
-import jieba
+import pyroid.jieba as jieba

 jieba.lcut("我来到北京清华大学")  # ['我', '来到', '北京', '清华大学']
 jieba.lcut("他来到了网易杭研大厦")  # ['他', '来到', '了', '网易', '杭研', '大厦']
```

jieba itself is not needed, except for the [delegated features](#delegated-features).

## Segmenting

The functions and their arguments are jieba's:

| Function | Returns |
|---|---|
| `cut(sentence, cut_all=False, HMM=True)` | a generator of words |
| `lcut(...)` | the same words as a list |
| `cut_for_search(sentence, HMM=True)` / `lcut_for_search(...)` | the words, each long word preceded by the 2- and 3-character dictionary words inside it |
| `tokenize(sentence, mode="default", HMM=True)` | `(word, start, end)` tuples with character offsets; `mode="search"` for the search words |
| `get_DAG(sentence)` | for each character, the end indices of the dictionary words that start there |

```python
import pyroid.jieba as jieba

jieba.lcut("我来到北京清华大学", cut_all=True)
# ['我', '来到', '北京', '清华', '清华大学', '华大', '大学']
jieba.lcut_for_search("小明硕士毕业于中国科学院计算所")
# ['小明', '硕士', '毕业', '于', '中国', '科学', '学院', '科学院', '中国科学院', '计算', '计算所']
list(jieba.tokenize("永和服装饰品有限公司"))
# [('永和', 0, 2), ('服装', 2, 4), ('饰品', 4, 6), ('有限公司', 6, 10)]
```

- `HMM=True` (the default) finds words that are not in the dictionary, such as 杭研 above, with
  jieba's hidden Markov model. `HMM=False` keeps to dictionary words.
- `cut_all=True` lists every dictionary word in the text, overlapping, as jieba does. That
  includes the empty strings jieba yields next to whitespace.
- Bytes are decoded as jieba decodes them: UTF-8, else GBK, ignoring errors.

The port follows jieba step by step. That covers the split into blocks with `re_han_default` and
`re_skip_default`, the prefix-dictionary DAG, and the maximum-probability route. The route uses
the same floating-point operations, so ties between equally probable routes go the same way. It
also covers the BMES Viterbi for unknown words.

## Many documents

A list or tuple of `str`, or an Arrow string column, gives one list of words per document. The
documents are segmented in parallel on pyroid's shared pool. Arrow nulls give `None`.

```python
import pyarrow as pa
import pyroid.jieba as jieba

docs = ["我来到北京清华大学", "他来到了网易杭研大厦"]
jieba.lcut(docs)  # [['我', '来到', '北京', '清华大学'], ['他', '来到', '了', '网易', '杭研', '大厦']]
jieba.lcut(docs, threads=4)  # at most 4 tasks at once; threads=1 runs on the calling thread
jieba.lcut_for_search(pa.array(["中国科学院", None]))  # [['中国', '科学', '学院', '科学院', '中国科学院'], None]
```

`cut_for_search` and `tokenize` take batches the same way. `threads=` is keyword-only. It
follows the [threading conventions](../concepts/threads.md), and on free-threaded CPython the
GIL stays off. As in jieba, `enable_parallel()` makes the module-level `cut` and
`cut_for_search` split a long text into lines and segment them in parallel. pyroid does this with
threads rather than processes, and it also works on Windows.

## Dictionaries

`load_userdict`, `add_word`, `del_word` and `suggest_freq` change the dictionary as they do in
jieba:

```python
import pyroid.jieba as jieba

jieba.lcut("李小福是创新办主任")  # ['李小福', '是', '创新', '办', '主任']
jieba.add_word("创新办", freq=3)
jieba.lcut("李小福是创新办主任")  # ['李小福', '是', '创新办', '主任']

jieba.lcut("如果放到post中将出错。", HMM=False)  # [..., 'post', '中将', '出错', '。']
jieba.suggest_freq(("中", "将"), tune=True)  # 494, now the frequency of 中将
jieba.lcut("如果放到post中将出错。", HMM=False)  # ['如果', '放到', 'post', '中', '将', '出错', '。']

jieba.del_word("创新办")
jieba.load_userdict("userdict.txt")  # one "word [freq] [tag]" per line
```

- `load_userdict` takes a path, or a file opened in binary mode (UTF-8; a leading BOM is skipped)
  or in text mode. Lines follow jieba's `re_userdict`: a word may contain spaces, the frequency is
  ASCII digits and the tag lowercase ASCII letters.
- `add_word(word)` without a frequency uses `suggest_freq(word)`, which keeps the word whole.
  `del_word` sets the frequency to 0, and also stops the HMM from producing the word, in every
  tokenizer, as jieba does.
- `Tokenizer(dictionary=path)` uses another main dictionary, in jieba's `dict.txt` format. It is
  read when first needed, and `set_dictionary` and `initialize(dictionary)` switch it. As in
  jieba, it is loaded at the first run of word characters (Chinese characters, ASCII letters and
  digits, `+#&._%-`), so text without one is cut without it.
  A broken dictionary file raises jieba's `ValueError: invalid dictionary entry in ... at Line N`.
- `dt` is the default tokenizer, and the module-level functions are its methods. `FREQ`, `total`
  and `user_word_tag_tab` are there too. `FREQ` is a read-only mapping of every word and word
  prefix to its frequency.

## Delegated features

These are handed to the real jieba 0.42.1, which must then be installed. The first call emits a
[`pyroid.FallbackWarning`](../concepts/fallbacks.md). Without jieba, the call raises
[`pyroid.PyroidError`](../reference/pyroid.md#pyroid.PyroidError).

- `pyroid.jieba.posseg`: part-of-speech tagging.
- `pyroid.jieba.analyse`: TF-IDF and TextRank keywords.
- Paddle mode: `enable_paddle()` and `use_paddle=True`. Unlike jieba, pyroid does not
  `pip install` paddlepaddle-tiny for you.
- `get_dict_file()` for the default dictionary, which pyroid stores in another form.

Before each delegated call, the words added through `pyroid.jieba` are replayed onto jieba's
tokenizer, in order, so posseg and analyse see them:

```python
import pyroid.jieba as jieba
import pyroid.jieba.posseg as pseg

jieba.add_word("创新办", tag="i")
pseg.lcut("李小福是创新办主任")
# [pair('李小福', 'nr'), pair('是', 'v'), pair('创新办', 'i'), pair('主任', 'b')], and once:
# FallbackWarning: pyroid.jieba: jieba.posseg is not ported to Rust; calling jieba 0.42.1
```

## Differences from jieba

- A `str` with a lone surrogate (`"\ud800"`) raises `UnicodeEncodeError`, as everywhere in
  pyroid. jieba treats it as a character.
- Frequencies, and the dictionary's total, must fit in a signed 64-bit integer. jieba takes any
  Python int.
- `cut` is a generator, as in jieba, but it segments the whole input on the first `next()`.
- `FREQ` cannot be written to directly; use `add_word`. Iterating it lists the keys in code point
  order, not in the dictionary file's order.
- Arguments of the wrong type raise `TypeError` (`pyroid.InputError` for the text), where jieba
  raises `AttributeError`.
- There is no `jieba.finalseg` module, and jieba's log messages ("Building prefix dict ...") are
  not printed.

## Parity

Every check found 0 differences from jieba 0.42.1:

| Check | Comparisons |
|---|--:|
| Parity suite against live jieba: FineWeb-2 Chinese (`cmn_Hani`) and the adversarial set in all 9 modes, the batch path, user dictionaries, `FREQ` | 387,557 |
| Replay of recorded jieba and datatrove outputs (Rust): jieba's own test sentences, FineWeb-2 excerpts, mixed scripts, emoji and every kind of whitespace, fuzz, user and custom dictionary scenarios | 73,236 |
| 20,000 FineWeb-2 `cmn_Hani` training documents (29.5 M characters), in 5 modes and the batch path | 120,000 documents × mode |

Hypothesis properties also compare every mode against live jieba, as well as sequences of
`add_word`, `del_word`, `suggest_freq` and `load_userdict`, broken and unusual dictionary files,
and every code point. The route uses `math.log` on integers. Its results are bit-identical on
CPython 3.12 to 3.15, and so are jieba's outputs. See [Compatibility](../compatibility.md#jieba).

## Speed

Docker/OrbStack VM on Apple Silicon (12 vCPU), indicative. These are 1,000 FineWeb-2 `cmn_Hani`
documents of about 1,450 characters each, on free-threaded CPython 3.15t with a release wheel.
Medians of 5 runs (jieba: 2), with every document's words checked against jieba's first. µs per
document:

| Run | `lcut` | `lcut_for_search` |
|---|--:|--:|
| jieba, one thread | 3,072 | 2,998 |
| jieba, 8 threads (free-threaded) | 2,245 | 2,499 |
| jieba, 8 processes | 584 | 895 |
| pyroid, `lcut(doc)` per document | 162 (19×) | 187 (16×) |
| pyroid, `lcut(docs, threads=1)` | 127 (24×) | 160 (19×) |
| pyroid, `lcut(docs, threads=8)` | 53 (58×) | 67 (45×) |

With 8 threads, pyroid's batch gains about 2.4× over one thread. Building the Python lists of
words (about 35 µs per document) runs on the calling thread, while the segmentation itself takes
about 90 µs per document per thread. A fresh interpreter takes 27 ms to reach its first segmented
sentence with pyroid (16 ms on CPython 3.13), import included, and about 410 ms with jieba. See
[Benchmarks](../benchmarks.md#other-modules-languages-and-integrations).

The default dictionary adds 2.5 MB to the extension module, and the HMM tables 0.24 MB. The wheel
grows by 1.5 MB.

API: [`pyroid.jieba` reference](../reference/jieba.md).
