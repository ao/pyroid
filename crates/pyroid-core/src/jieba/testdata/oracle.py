"""Record what jieba 0.42.1 returns, for the replay test of `pyroid_core::jieba`.

Derived from jieba 0.42.1 (MIT), © 2013 Sun Junyi, and datatrove 0.10.1 (Apache-2.0),
© HuggingFace; modified. The FineWeb-2 excerpts in the goldens are from
HuggingFaceFW/fineweb-2 (ODC-By 1.0), subset cmn_Hani.

`pyroid_core::jieba` must cut exactly as jieba 0.42.1 does. This script runs jieba (and
datatrove's Chinese word tokenizer: spaCy's `ChineseTokenizer` over jieba, plus the
sentencizer) on fixed inputs and writes what they return to `golden-*.jsonl` next to it;
`../replay.rs` replays every row through the Rust API and requires identical results.

Run inside the pyroid-dev container (never on the host) with an interpreter that has
jieba==0.42.1, plus datatrove==0.10.1, spacy==3.8.18 and pyarrow for a full recording:

    python oracle.py record DIR --jieba-src SDIST --readme README.md \\
        --fineweb cmn_Hani.parquet [--dict-small dict.txt.small]
    python oracle.py record DIR --texts-from DIR0 [--jieba-only] [--dict-small ...]
    python oracle.py compare DIR1 DIR2 [--jieba-only]
    python oracle.py logcheck [dict.txt]

`record` writes the goldens and `recorded.json` (versions, sources, row counts, sha256 of
each golden) into DIR. `--jieba-src` is the unpacked jieba-0.42.1 sdist (its `test/*.py`),
`--readme` jieba's README.md (both only for corpus `tests`); `--texts-from` takes the texts
of the corpora `tests`, `fineweb` and `mixed` from an earlier recording instead, so no
sdist or parquet reader is needed (the fuzz texts and the scenarios are generated here).
`--jieba-only` leaves out datatrove's fields (for interpreters without datatrove). `compare`
reports whether two recordings are identical (`--jieba-only`: ignoring datatrove's fields).
`logcheck` checks that `math.log(n) == math.log(float(n))` bit for bit for every frequency
of `dict.txt` and its total (jieba's `calc` takes `log` of ints).

The goldens are deterministic: the same interpreter gives the same bytes on every run.

Corpora (one text per row): `golden-tests.jsonl` holds every string literal of jieba's
`test/*.py` and README code passed to a cut function or containing a CJK ideograph;
`golden-fineweb-*.jsonl` FineWeb-2 paragraphs (split on "\\n", cut at 400 characters,
deduplicated, a seeded sample); `golden-mixed.jsonl` such paragraphs with Latin words,
numbers, URLs, punctuation, traditional characters, emoji, CJK outside jieba's range,
combining marks and whitespace inserted, plus handwritten lines; `golden-fuzz.jsonl` seeded
random strings. A row is

    {"t": text, "c": cut(t), "c0": cut(t, HMM=False), "a": cut(t, cut_all=True),
     "s": cut_for_search(t), "s0": cut_for_search(t, HMM=False),
     "zd" or "zw": datatrove's word_tokenize(t), "zs": datatrove's span_tokenize(t)}

plus "k" (what the text is) or "d" (the FineWeb-2 row it comes from). Word lists are
written compactly (see `code`): the positions in "c", "c0", "s" and "s0" are those of
`tokenize(t, mode, HMM)` (whose words are checked to be the cut's), so each of those fields
records both the cut and the tokenize output. "zd" lists the indices of the words of "c"
that datatrove's words leave out (its words are the rest), "zw" holds datatrove's words
when they are not such a sub-list, and "zs" the sentence spans (character offsets)
flattened. datatrove's fields are only in the rows of a subset of the fuzz texts.

`golden-scenarios.jsonl` (and `golden-dict-small.jsonl`, for jieba's
`extra_dict/dict.txt.small`) records dictionary changes step by step, one row per step:

    {"op": "new", "scn": name, "dict": "default" | file, "len": len(FREQ), "total": total,
     ["entries": [[word, freq], ...] in file order, "items": sorted(FREQ.items()),
      "src": the file's text], ["keep_fs": true]}
    {"op": "add", "w": word, "f": int | null, ["tag": tag], ["via": "load_userdict"],
     "freq": FREQ[word] afterwards | "raise": exception, "len", "total", "get"}
    {"op": "del", "w": word, "len", "total", "get"}
    {"op": "sf", "seg": str | [str, ...], "tune": bool, "ret": int | "raise": exception,
     ["len", "total", "get"]}
    {"op": "probe", "t": text, "c", "c0", "a", "s", "s0"}

"get" is `[[w, FREQ.get(w)], ...]` for the word and its prefixes. Every scenario starts
with a fresh `jieba.Tokenizer()` and an empty `finalseg.Force_Split_Words` (process-global
in jieba; `add_word(w, 0)` and `del_word(w)` add `w` to it), unless "keep_fs".

A word list is a string of items, one per word: a digit `n` (or `(n)` for n >= 10) is the
next `n` characters after the previous word (from 0 for the first); `<n>` the `n`
characters at the previous word's start; `<d,n>` the `n` characters `d` characters after
the previous word's start (`d` may be negative). A string "=X" is the same list as field
X of the row, a JSON list holds the words themselves (when some word is not a substring of
the text), and {"raise": name} records the exception the call raised. Every encoded list is
decoded and compared with what jieba returned before it is written.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import hashlib
import itertools
import json
import math
import re
import struct
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

# Look-alike characters on purpose.
# ruff: noqa: RUF001

JIEBA_VERSION = "0.42.1"
SEED = 20261107
N_FINEWEB = 2400  # FineWeb-2 paragraphs, over two files
N_MIXED = 320  # FineWeb-2 paragraphs with insertions
N_FUZZ = 2000
FUZZ_DATATROVE_EVERY = 3  # datatrove's fields for every third fuzz text
MAX_PARAGRAPH = 400
DICT_SMALL_SHA256 = "479cf8cc37e78bc908ae330d9153375547d4ff7f7d03a1ac4d4908d3677ff664"
DATATROVE_FIELDS = ("zd", "zw", "zs")
CUT_CALLS = frozenset(
    {"cuttest", "cut", "lcut", "cut_for_search", "lcut_for_search", "tokenize", "analyzer"}
)
HAN_ANY = re.compile("[\u3400-\u9fff\uf900-\ufaff\U00020000-\U0003134f]")


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


# --------------------------------------------------------------------------
# Word list encoding


def code(text: str, words: Sequence[str], starts: Sequence[int] | None = None) -> str | list[str]:
    """`words` (substrings of `text`, at `starts` if given) in the compact form."""
    items: list[str] = []
    ps = pe = 0
    for i, w in enumerate(words):
        n = len(w)
        if starts is not None:
            s = starts[i]
            if text[s : s + n] != w:
                raise AssertionError(f"{w!r} is not at {s} in {text!r}")
        elif text.startswith(w, pe):
            s = pe
        else:
            found = nearest(text, w, ps)
            if found is None:
                return list(words)
            s = found
        if s == pe:
            items.append(str(n) if n < 10 else f"({n})")
        else:
            d = s - ps
            items.append(f"<{n}>" if d == 0 else f"<{d},{n}>")
        ps, pe = s, s + n
    return "".join(items)


def nearest(text: str, w: str, ps: int) -> int | None:
    """The start of the occurrence of `w` in `text` closest to `ps`."""
    best = None
    i = text.find(w)
    while i != -1:
        if best is None or abs(i - ps) < abs(best - ps):
            best = i
        if i > ps:
            break
        i = text.find(w, i + 1)
    return best


ITEM = re.compile(r"(\d)|\((\d+)\)|<(-?\d+),(\d+)>|<(\d+)>")


def decode(text: str, value: str | list[str]) -> list[tuple[str, int | None]]:
    """(word, start) for each item of an encoded list (start None for literal words)."""
    if isinstance(value, list):
        return [(w, None) for w in value]
    out: list[tuple[str, int | None]] = []
    ps = pe = 0
    pos = 0
    while pos < len(value):
        m = ITEM.match(value, pos)
        if m is None:
            raise AssertionError(f"bad code {value!r} at {pos}")
        pos = m.end()
        if m.group(1) or m.group(2):
            s, n = pe, int(m.group(1) or m.group(2))
        elif m.group(5):
            s, n = ps, int(m.group(5))
        else:
            s, n = ps + int(m.group(3)), int(m.group(4))
        if not 0 <= s <= s + n <= len(text):
            raise AssertionError(f"code {value!r} leaves the text")
        out.append((text[s : s + n], s))
        ps, pe = s, s + n
    return out


def checked_code(text: str, words: Sequence[str], starts: Sequence[int] | None = None) -> Any:
    v = code(text, words, starts)
    got = decode(text, v)
    if [w for w, _ in got] != list(words):
        raise AssertionError(f"code {v!r} does not round-trip for {text!r}")
    if starts is not None and [s for _, s in got] != list(starts):
        raise AssertionError(f"code {v!r} loses positions for {text!r}")
    return v


def dumps(row: dict[str, Any]) -> str:
    s = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    # Only possible inside JSON strings, where the escapes mean the same; every line
    # splitter then agrees on where the rows end.
    return s.replace("\x85", "\\u0085").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


# --------------------------------------------------------------------------
# Recording jieba


def _call(fn: Callable[[], list[Any]]) -> tuple[list[Any] | None, str | None]:
    try:
        return fn(), None
    except Exception as e:
        return None, type(e).__name__


def cut_fields(tok: Any, text: str) -> dict[str, Any]:
    """jieba's five cuts of `text` (and the four tokenize outputs), encoded."""
    specs: list[tuple[str, Callable[[], list[Any]], Callable[[], list[Any]] | None]] = [
        ("c", lambda: tok.lcut(text), lambda: list(tok.tokenize(text))),
        ("c0", lambda: tok.lcut(text, HMM=False), lambda: list(tok.tokenize(text, HMM=False))),
        ("a", lambda: tok.lcut(text, cut_all=True), None),
        ("s", lambda: tok.lcut_for_search(text), lambda: list(tok.tokenize(text, mode="search"))),
        (
            "s0",
            lambda: tok.lcut_for_search(text, HMM=False),
            lambda: list(tok.tokenize(text, mode="search", HMM=False)),
        ),
    ]
    out: dict[str, Any] = {}
    for key, cut, tokenize in specs:
        words, err = _call(cut)
        if tokenize is None:
            out[key] = {"raise": err} if words is None else checked_code(text, words)
            continue
        toks, terr = _call(tokenize)
        if err != terr:
            raise AssertionError(f"{key}: cut raised {err}, tokenize {terr}, for {text!r}")
        if words is None or toks is None:
            out[key] = {"raise": err}
            continue
        if [w for w, _, _ in toks] != words or any(e != s + len(w) for w, s, e in toks):
            raise AssertionError(f"{key}: tokenize's words are not the cut's for {text!r}")
        out[key] = checked_code(text, words, [s for _, s, _ in toks])
    # The same code as an earlier field: a reference to it.
    seen: dict[str, str] = {}
    for key in list(out):
        v = out[key]
        if isinstance(v, str):
            if v in seen:
                out[key] = "=" + seen[v]
            else:
                seen[v] = key
    return out


def drop_indices(words: Sequence[str], kept: Sequence[str]) -> list[int] | None:
    """Indices of `words` whose removal leaves `kept`, or None."""
    j = 0
    drop = []
    for i, w in enumerate(words):
        if j < len(kept) and w == kept[j]:
            j += 1
        else:
            drop.append(i)
    return drop if j == len(kept) else None


class Datatrove:
    """datatrove's word tokenizer for cmn_Hani (SpaCyTokenizer("zh"))."""

    def __init__(self) -> None:
        from datatrove.utils.word_tokenizers import load_word_tokenizer

        self.tok = load_word_tokenizer("cmn_Hani")

    def fields(self, text: str, cut: Sequence[str]) -> dict[str, Any]:
        words = self.tok.word_tokenize(text)
        spans = self.tok.span_tokenize(text)
        out: dict[str, Any] = {}
        drop = drop_indices(cut, words)
        if drop is not None:
            out["zd"] = drop
        else:
            out["zw"] = checked_code(text, words)
        out["zs"] = [x for s, e in spans for x in (s, e)]
        return out


Item = tuple[str, dict[str, Any]]


def record_rows(
    items: Iterable[Item], dt: Datatrove | None, with_dt: Callable[[int], bool]
) -> list[str]:
    import jieba
    import jieba.finalseg

    rows = []
    for i, (text, extra) in enumerate(items):
        if jieba.finalseg.Force_Split_Words:
            raise AssertionError("Force_Split_Words is not empty")
        row: dict[str, Any] = {**extra, "t": text}
        row.update(cut_fields(jieba.dt, text))
        if dt is not None and with_dt(i) and "IS_ALPHA" not in text:
            row.update(dt.fields(text, jieba.lcut(text)))
        rows.append(dumps(row))
    return rows


# --------------------------------------------------------------------------
# Corpus: jieba's own test inputs


def _callee(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def literal_inputs(tree: ast.AST) -> list[str]:
    """String literals passed first to a cut-like call, or containing a CJK ideograph."""
    found: dict[int, ast.Constant] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and _callee(node) in CUT_CALLS
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            found[id(node.args[0])] = node.args[0]
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and HAN_ANY.search(node.value)
        ):
            found[id(node)] = node
    nodes = sorted(found.values(), key=lambda n: (n.lineno, n.col_offset))
    return [n.value for n in nodes]


def readme_blocks(text: str) -> Iterator[str]:
    """The python and pycon code blocks of a Markdown file, as Python source."""
    for m in re.finditer(r"^```(python|pycon)\n(.*?)^```", text, re.M | re.S):
        body = m.group(2)
        if m.group(1) == "pycon":
            body = "\n".join(
                line[4:] for line in body.splitlines() if line.startswith((">>> ", "... "))
            )
        yield body


def parse_any(source: str) -> list[ast.AST]:
    """`source` parsed whole, else line by line (skipping lines that do not parse)."""
    try:
        return [ast.parse(source)]
    except SyntaxError:
        trees = []
        for line in source.splitlines():
            try:
                trees.append(ast.parse(line.strip()))
            except SyntaxError:
                continue
        return trees


def test_items(sdist: Path, readme: Path | None) -> list[Item]:
    seen: set[str] = set()
    items: list[Item] = []
    sources: list[tuple[str, str]] = [
        (p.name, p.read_text(encoding="utf-8")) for p in sorted((sdist / "test").glob("*.py"))
    ]
    if readme is not None:
        sources += [("README.md", b) for b in readme_blocks(readme.read_text(encoding="utf-8"))]
    for name, source in sources:
        for tree in parse_any(source):
            for s in literal_inputs(tree):
                if s not in seen:
                    seen.add(s)
                    items.append((s, {"k": name}))
    return items


# --------------------------------------------------------------------------
# Corpus: FineWeb-2 paragraphs


def fineweb_paragraphs(parquet: Path) -> list[tuple[str, int]]:
    """(paragraph, row) for every distinct paragraph, in file order."""
    import pyarrow.parquet as pq

    texts = pq.read_table(parquet, columns=["text"], use_threads=False).column("text").to_pylist()
    seen: set[str] = set()
    out = []
    for row, text in enumerate(texts):
        for p in text.split("\n"):
            p = p[:MAX_PARAGRAPH]
            if p and p not in seen:
                seen.add(p)
                out.append((p, row))
    return out


def pick(rng: Any, n: int, k: int) -> list[int]:
    """`k` distinct indices below `n`, sorted (a partial Fisher-Yates shuffle)."""
    idx = list(range(n))
    k = min(k, n)
    for i in range(k):
        j = i + rng.randrange(n - i)
        idx[i], idx[j] = idx[j], idx[i]
    return sorted(idx[:k])


# --------------------------------------------------------------------------
# Corpus: mixed scripts

# fmt: off
WHITESPACE = [
    "\r\n", "\n\r", "\t", "\n", "\r", " ", "  ", "\u3000", "\xa0", "\u1680", "\u2000", "\u2002",
    "\u2003", "\u2009", "\u200a", "\u2028", "\u2029", "\u202f", "\u205f", "\x0b", "\x0c", "\x1c",
    "\x1d", "\x1e", "\x1f", "\x85", " \n ", "\n\n", "\r\n\r\n", "\t\t", "\r\r\n", " \u3000 ",
]
# Not whitespace for Python (nor for jieba's \s).
NOT_WHITESPACE = ["\u200b", "\u180e", "\ufeff", "\u2060", "\u200d", "\u200c"]
LATIN = [
    "Python", "iPhone", "WiFi", "OpenAI", "the", "Hello world", "ChatGPT", "GPT-4", "macOS",
    "Linux", "x", "A", "ok", "USB", "CEO", "NBA", "DNA", "app", "PK", "VIP", "naïve", "Zürich",
    "Ελλάδα", "Москва", "café",
]
NUMBERS = [
    "3.14%", "1.5e3", "2024", "2024年", "100", "0.5", "1,000", "50%", "-3", "+86", "3.14", "12:30",
    "2024-01-01", "v1.2.3", "１２３", "３．１４％", "1e-9", "007", "10%-20%", "1/2", "²", "½", "①",
    "Ⅷ", "〇", "零点五", "٣", "१२", "3.", ".5", "1..2", "99.9%%",
]
TECH = [
    "C++", "c++", "C#", "c#", "AT&T", "at&t", "e-mail", "E-mail", ".NET", "Node.js", "a_b",
    "x86_64", "U.S.", "Wi-Fi", "B超", "T恤", "X光", "Q币", "A股", "K歌", "C语言", "PM2.5", "H1N1",
    "Ｃ＋＋", "F#", "&amp;", "#hashtag", "_init_", "--", "...", "%%", "+-", "a.b.c", "1.2.3.4",
    "C++11", "c#5", "AT&T公司", "e-mail地址",
]
URLS = [
    "https://example.com/a?b=1&c=2", "http://www.baidu.com", "www.qq.com", "user@example.com",
    "http://中文.com/路径?q=测试", "ftp://192.168.1.1:21/", "https://zh.wikipedia.org/wiki/北京",
    "mailto:a.b@c.cn", "#锚点", "https://t.co/AbC_12-x%20y",
]
ASCII_PUNCT = list("!\"$'()*,/:;<=>?@[\\]^`{|}~")
FULL_PUNCT = [
    *"，。、；：？！“”‘’（）《》【】…—·「」『』～￥％＃＆＋－．＿〈〉〔〕｛｝｜＠", "……", "——",
]
TRADITIONAL = [
    "繁體中文", "臺灣", "這個", "電腦軟體", "資訊", "網際網路", "國語", "學習", "愛", "說話", "龍",
    "鬱", "體驗", "後來", "裡面", "著名", "頭髮", "乾燥", "臺北市", "香港特別行政區",
]
EMOJI = [
    "😀", "👍", "👍🏽", "❤️", "❤", "👨‍👩‍👧", "👩‍💻", "🏳️‍🌈", "🇨🇳", "🇹🇼", "🇭🇰", "1️⃣", "#️⃣",
    "🀄", "🈶", "㊗️", "©", "™", "🐉", "🧧",
    "🏴\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f",
]
# CJK characters outside jieba's [\u4E00-\u9FD5] (and its two ends, inside it).
OUTSIDE = [
    "\u9fd6", "\u9fe0", "\u9fea", "\u9fef", "\u9fff", "\u3400", "\u4db5", "\u4dbf", "\U00020000",
    "\U00020bb7", "\U0002a6a5", "\U0002b740", "\U00030000", "\uf900", "\ufa0e", "\ufa6d",
    "\U0002f800", "\u2e80", "\u2f00", "\u3005", "\u3007", "\u303b", "\u31c0", "\u4dc0", "\u4e00",
    "\u9fd5", "\u9fa5", "\u4e00\u9fd5", "\u9fd5\u9fd6",
]
MARKS = [
    "e\u0301", "a\u0308", "\u0301", "\u20dd", "\ufe0f", "\ufe0e", "\u200d", "\u200b", "\u200c",
    "\ufeff", "\u2060", "\u180e", "中\u0301", "\u302a", "\u3099", "\u0300\u0301",
]
MIX_POOLS: list[tuple[list[str], float]] = [
    (LATIN, 3), (NUMBERS, 3), (TECH, 3), (URLS, 1), (ASCII_PUNCT, 2), (FULL_PUNCT, 2),
    (TRADITIONAL, 2), (EMOJI, 2), (OUTSIDE, 2), (MARKS, 1), (WHITESPACE, 3),
    (NOT_WHITESPACE, 0.5),
]
TRADITIONAL_SENTENCES = [
    "我們今天去臺北看電影。",
    "這個軟體的介面很漂亮，但是速度太慢了。",
    "香港特別行政區政府宣佈新的經濟政策。",
    "臺灣的夜市文化非常有名，吸引許多觀光客。",
    "他說：「明天會下雨嗎？」",
    "資訊工程學系的學生正在準備期末考試。",
    "請問捷運站怎麼走？",
    "這家餐廳的牛肉麵很好吃，價格也很合理。",
    "電腦螢幕上顯示著一則重要的訊息。",
    "澳門的葡式蛋撻遠近馳名。",
    "「台中」正確應該不會被切開。",
    "又跛又啞",
]
HANDWRITTEN = [
    "", " ", "\r\n", "中", "a", "。", "😀", "1", "-", "%", ".", "+#&._%-", "中文", "abc", "ABC",
    "價格上漲了3.14%，約1.5e3元。",
    "我用C++和c#寫程式，也用Python。",
    "AT&T公司的e-mail是support@att.com",
    "網址：https://www.example.com/path?x=1&y=2。",
    "下载速度达到100Mbps（约12.5MB/s）",
    "温度是-3.5℃，湿度80%",
    "ＡＢＣ全角字母和１２３全角数字",
    "iPhone 15 Pro Max售价7999元",
    "第1名、第2名和第3名",
    "2024年1月1日星期一",
    "Windows 11和macOS 14",
    "H1N1病毒和PM2.5",
    "价格：￥100.00/个",
    "C++11、C++14与C++17",
    "a.b.c.d",
    "foo_bar_baz是变量名",
    "#话题# @用户",
    "  前导空格",
    "结尾空格  ",
    "\t制表符\t",
    "换行\r\n回车",
    "中\u3000文",
    "中\xa0文",
    "👨‍👩‍👧中文",
    "中🇨🇳国",
    "我爱北京天安门😀😀",
    "B超和T恤和X光",
    "3.14.15.926",
    "1,234,567.89元",
    "50%-60%的人",
    "e-mail和E-MAIL和Email",
    "c++和C++和Ｃ＋＋",
    "\u4e00\u9fd5\u9fd6\u9fff",
    "鿕鿖鿗",
    "𠮷野家",
    "㐀㐁㐂中文",
    "豈更車",
    "中\u0301文\u0308",
    "繁體中文與简体中文混合使用",
    "ｈｔｔｐ：／／ｅｘａｍｐｌｅ．ｃｏｍ",
    "\x1c\x1d\x1e\x1f",
    "\x85中\x85",
    "\u2028段落\u2029",
    "...。。。",
    "中文.中文",
    "中文_中文-中文",
    "abc中文def",
    "123中文456",
    "a1b2c3中d4",
    "%%%中文%%%",
    "&&&AT&&T",
    "---",
    "___",
    "...",
    "+++",
    "###",
    "a-b-c-d-e-f-g-h-i-j-k-l-m-n-o-p",
    "0123456789" * 3,
    "x" * 60,
    "中" * 30,
    "的" * 20,
    "了了了了了",
    "一一一一一一",
]
# fmt: on


class Weighted:
    """A weighted choice that only draws `random()` (stable across CPython versions)."""

    def __init__(self, rng: Any, choices: Sequence[tuple[Any, float]]) -> None:
        self.rng = rng
        self.values = [v for v, _ in choices]
        self.cum = []
        total = 0.0
        for _, w in choices:
            total += w
            self.cum.append(total)

    def __call__(self) -> Any:
        x = self.rng.random() * self.cum[-1]
        return self.values[min(bisect.bisect_right(self.cum, x), len(self.values) - 1)]


def mixed_items(paragraphs: Sequence[str], seed: int) -> list[Item]:
    import random

    rng = random.Random(seed)
    items: list[Item] = [(t, {"k": "hand"}) for t in HANDWRITTEN]
    items += [(t, {"k": "trad"}) for t in TRADITIONAL_SENTENCES]
    for w in WHITESPACE + NOT_WHITESPACE:
        for tmpl in (
            "中文{0}English{0}123{0}。",
            "{0}前导",
            "结尾{0}",
            "{0}",
            "a{0}b",
            "我{0}{0}你",
        ):
            items.append((tmpl.format(w), {"k": "ws"}))
        items.append((f"C++{w}AT&T{w}3.14%", {"k": "ws"}))
    for c in OUTSIDE:
        for tmpl in ("中{0}文", "{0}{0}", "北京{0}大学", "a{0}1"):
            items.append((tmpl.format(c), {"k": "cjk"}))
    for e in EMOJI + MARKS:
        items.append((f"我爱{e}北京", {"k": "emoji"}))
    base = [p for p in paragraphs if 8 <= len(p) <= 150]
    pool = Weighted(rng, MIX_POOLS)
    for i in pick(rng, len(base), N_MIXED):
        t = base[i]
        for _ in range(rng.randint(1, 4)):
            choices = pool()
            item = choices[rng.randrange(len(choices))]
            pos = rng.randint(0, len(t))
            t = t[:pos] + item + t[pos:]
        items.append((t, {"k": "mix"}))
    seen: set[str] = set()
    out = []
    for t, extra in items:
        if t not in seen:
            seen.add(t)
            out.append((t, extra))
    return out


# --------------------------------------------------------------------------
# Corpus: fuzz

FUZZ_WS = ["\r\n", "\n\r", "\t", "\n", "\r", " ", "  ", "\u3000", "\xa0", "\u2028", "\x1c", "\x85"]
FUZZ_PUNCT = [*ASCII_PUNCT, *"，。、；：？！“”（）《》…—「」"]
FUZZ_EMOJI = ["😀", "👍🏽", "👨‍👩‍👧", "🇨🇳", "❤️", "1️⃣"]
FUZZ_OUTSIDE = ["\u9fd6", "\u9fff", "\u3400", "\U00020bb7", "\uf900", "\u3007"]


class Fuzzer:
    """Random strings over dictionary characters and words, ASCII, whitespace and symbols."""

    def __init__(self, seed: int, freq: dict[str, int]) -> None:
        import random

        self.rng = random.Random(seed)

        def han(w: str) -> bool:
            return all("\u4e00" <= c <= "\u9fd5" for c in w)

        singles = sorted(
            ((f, w) for w, f in freq.items() if len(w) == 1 and f > 0 and han(w)),
            key=lambda x: (-x[0], x[1]),
        )
        self.common = [w for _, w in singles[:400]]
        long_words = sorted(w for w, f in freq.items() if f > 0 and len(w) >= 5 and han(w))
        step = max(1, len(long_words) // 300)
        self.long_words = long_words[::step][:300]
        self.long_chars = sorted({c for w in self.long_words for c in w})
        self.ascii_words = sorted(
            w for w, f in freq.items() if f > 0 and any(c.isascii() for c in w)
        )
        self.units = Weighted(
            self.rng,
            [
                (lambda: self.pick(self.common), 30),
                (lambda: self.pick(self.long_chars), 10),
                (lambda: self.pick(self.long_words), 4),
                (self.long_prefix, 4),
                (self.long_suffix, 2),
                (lambda: self.pick(self.ascii_words), 3),
                (lambda: self.pick("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"), 8),
                (lambda: self.pick("0123456789"), 6),
                (self.ascii_run, 3),
                (self.number, 3),
                (lambda: self.pick("+#&._%-"), 6),
                (lambda: self.pick(FUZZ_WS), 6),
                (lambda: self.pick(FUZZ_PUNCT), 7),
                (lambda: self.pick(FUZZ_EMOJI), 2),
                (lambda: self.pick(FUZZ_OUTSIDE), 2),
                (lambda: self.pick(["\u0301", "\ufe0f", "\u200d", "\u200b"]), 1),
            ],
        )

    def pick(self, seq: Sequence[str]) -> str:
        return seq[self.rng.randrange(len(seq))]

    def long_prefix(self) -> str:
        w = self.pick(self.long_words)
        return w[: self.rng.randint(2, len(w) - 1)]

    def long_suffix(self) -> str:
        w = self.pick(self.long_words)
        return w[self.rng.randint(1, len(w) - 2) :]

    def ascii_run(self) -> str:
        return "".join(
            self.pick("abcdefghijklmnopqrstuvwxyz0123456789+#&._%-")
            for _ in range(self.rng.randint(2, 8))
        )

    def number(self) -> str:
        d = "".join(self.pick("0123456789") for _ in range(self.rng.randint(1, 4)))
        return d + self.pick(["", "%", ".5", ".25%", "e3", ",000", "."])

    def length(self) -> int:
        r = self.rng.random()
        if r < 0.05:
            return self.rng.randint(0, 2)
        if r < 0.6:
            return self.rng.randint(1, 25)
        if r < 0.92:
            return self.rng.randint(25, 50)
        return self.rng.randint(50, 80)

    def text(self) -> str:
        n = self.length()
        s = ""
        while len(s) < n:
            s += self.units()()
        return s[:n]


def fuzz_items(seed: int, n: int) -> list[Item]:
    import jieba

    jieba.dt.check_initialized()
    f = Fuzzer(seed, jieba.dt.FREQ)
    return [(f.text(), {}) for _ in range(n)]


# --------------------------------------------------------------------------
# Scenarios: dictionary changes

# jieba 0.42.1 test/userdict.txt (MIT), byte for byte (a BOM, a word with a space, a tag
# without a frequency).
USERDICT = (
    "\ufeff云计算 5\n李小福 2 nr\n创新办 3 i\neasy_install 3 eng\n好用 300\n韩玉赏鉴 3 nz\n"
    "八一双鹿 3 nz\n台中\n凱特琳 nz\nEdu Trust认证 2000\n"
).encode()

# Small dictionaries for `jieba.Tokenizer(dictionary=path)` (`gen_pfdict` reads them).
CUSTOM_DICTS: dict[str, bytes] = {
    # Duplicates (the last frequency wins, the total counts both), a frequency of 0, a word
    # that is a prefix of another, one that is first a prefix and then a word, non-CJK words.
    "basic.txt": (
        "北京 100 ns\n北京大学 20 nt\n清华大学 15 nt\n清华 3 nr\n清华 7 nr\n大学 50 n\n"
        "中华人民共和国 2 ns\n中华 10 nz\n人民 40 n\n共和国 8 n\n我 500 r\n爱 200 v\n来到 30 v\n"
        "的 1000 uj\n了 0 ul\n天安门 12 ns\nC++ 5 nz\nAT&T 3 nz\ne-mail 2 n\n3.14 1 m\n"
        "hello 10 eng\n"
        "ab 2 x\nabc 1 x\nＡＢ 4 x\n😀 6 x\n臺灣 9 ns\n凱特琳 2 nr\n𠮷野家 3 nt\n一 80 m\n"
    ).encode(),
    # What `line.strip().decode()`, `split(' ')[:2]` and `int()` make of odd lines: a BOM
    # (kept in the first word), CRLF, extra columns, signs, underscores, leading zeros,
    # non-ASCII digits and trailing Unicode spaces.
    "quirks.txt": (
        "\ufeff首词 5 n\r\n第二 +7 m\r\n  前后空格 3 n  \r\n千 1_000 m\n零零七 007 m\n"
        "全角 ５ n\n阿拉伯 ٣ n\n空白 4\u3000\n多列 2 n extra columns\n负零 -0 x\n"
    ).encode(),
    # A negative frequency (`log` of it raises when the word is on a route).
    "negative.txt": "中文 -3 n\n中 10 n\n文 10 n\n英文 5 n\n".encode(),
    # Every frequency 0: a total of 0 (`log(0)` raises for every word block).
    "zero.txt": "中文 0 n\n中 0 n\nabc 0 x\n".encode(),
    "empty.txt": b"",
}


def file_lines(data: bytes) -> list[bytes]:
    """The lines of a file opened in binary mode, as iterating over it gives them."""
    parts = data.split(b"\n")
    return [p + b"\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def parse_dict(data: bytes) -> list[tuple[str, int]]:
    """(word, freq) per line, as `Tokenizer.gen_pfdict` reads a dictionary file."""
    entries = []
    for line in file_lines(data):
        text = line.strip().decode("utf-8")
        word, freq = text.split(" ")[:2]
        entries.append((word, int(freq)))
    return entries


def pfdict(entries: Iterable[tuple[str, int]]) -> tuple[dict[str, int], int]:
    freq: dict[str, int] = {}
    total = 0
    for word, f in entries:
        freq[word] = f
        total += f
        for ch in range(len(word)):
            freq.setdefault(word[: ch + 1], 0)
    return freq, total


def prefixes(w: str) -> list[str]:
    return [w[: i + 1] for i in range(len(w))] or [""]


class Scenarios:
    """Steps on one `jieba.Tokenizer` at a time, recorded as rows."""

    def __init__(self, dict_dir: Path) -> None:
        self.dict_dir = dict_dir
        self.rows: list[str] = []
        self.tok: Any = None
        self.probes: list[str] = []

    def emit(self, row: dict[str, Any]) -> None:
        self.rows.append(dumps(row))

    def state(self, words: Iterable[str] = ()) -> dict[str, Any]:
        out: dict[str, Any] = {"len": len(self.tok.FREQ), "total": self.tok.total}
        queries: list[str] = []
        for w in words:
            queries += [q for q in prefixes(w) if q not in queries]
        if queries:
            out["get"] = [[q, self.tok.FREQ.get(q)] for q in queries]
        return out

    def new(
        self, name: str, probes: Sequence[str], dictionary: str = "default", keep_fs: bool = False
    ) -> None:
        import jieba
        import jieba.finalseg

        if not keep_fs:
            jieba.finalseg.Force_Split_Words.clear()
        row: dict[str, Any] = {"op": "new", "scn": name, "dict": dictionary}
        if keep_fs:
            row["keep_fs"] = True
        if dictionary == "default":
            self.tok = jieba.Tokenizer()
        elif dictionary == "dict.txt.small":
            raise AssertionError("use new_small")
        else:
            data = CUSTOM_DICTS[dictionary]
            path = self.dict_dir / dictionary
            path.write_bytes(data)
            self.tok = jieba.Tokenizer(dictionary=str(path))
            self.tok.tmp_dir = str(self.dict_dir)
            self.tok.check_initialized()
            entries = parse_dict(data)
            if pfdict(entries) != (self.tok.FREQ, self.tok.total):
                raise AssertionError(f"{dictionary}: gen_pfdict read it differently")
            row["src"] = data.decode("utf-8")
            row["entries"] = [list(e) for e in entries]
            row["items"] = [list(kv) for kv in sorted(self.tok.FREQ.items())]
        self.tok.check_initialized()
        row.update(self.state())
        self.emit(row)
        self.probes = list(probes)
        self.probe()

    def new_small(self, name: str, path: Path, probes: Sequence[str]) -> None:
        import jieba
        import jieba.finalseg

        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha != DICT_SMALL_SHA256:
            raise SystemExit(f"{path}: sha256 {sha}, expected {DICT_SMALL_SHA256}")
        jieba.finalseg.Force_Split_Words.clear()
        self.tok = jieba.Tokenizer(dictionary=str(path))
        self.tok.tmp_dir = str(self.dict_dir)
        self.tok.check_initialized()
        if pfdict(parse_dict(data)) != (self.tok.FREQ, self.tok.total):
            raise AssertionError("dict.txt.small: gen_pfdict read it differently")
        row: dict[str, Any] = {"op": "new", "scn": name, "dict": "dict.txt.small", "sha256": sha}
        row["lines"] = len(file_lines(data))
        row.update(self.state())
        self.emit(row)
        self.probes = list(probes)
        self.probe()

    def probe(self, texts: Sequence[str] | None = None) -> None:
        for t in self.probes if texts is None else texts:
            self.emit({"op": "probe", "t": t, **cut_fields(self.tok, t)})

    def add(self, w: str, f: int | None = None, probe: bool = True) -> None:
        row: dict[str, Any] = {"op": "add", "w": w, "f": f}
        try:
            self.tok.add_word(w, f)
        except Exception as e:
            row["raise"] = type(e).__name__
        else:
            row["freq"] = self.tok.FREQ[w]
        row.update(self.state([w]))
        self.emit(row)
        if probe:
            self.probe()

    def delete(self, w: str, probe: bool = True) -> None:
        self.tok.del_word(w)
        self.emit({"op": "del", "w": w, **self.state([w])})
        if probe:
            self.probe()

    def sf(self, seg: str | tuple[str, ...], tune: bool, probe: bool = True) -> None:
        row: dict[str, Any] = {"op": "sf", "seg": seg if isinstance(seg, str) else list(seg)}
        row["tune"] = tune
        try:
            row["ret"] = self.tok.suggest_freq(seg, tune)
        except Exception as e:
            row["raise"] = type(e).__name__
        if tune:
            row.update(self.state([seg if isinstance(seg, str) else "".join(seg)]))
        self.emit(row)
        if tune and probe:
            self.probe()

    def userdict(self, name: str) -> None:
        """`load_userdict`: one "add" row per `add_word` call it makes."""
        path = self.dict_dir / name
        path.write_bytes(USERDICT)
        orig = self.tok.add_word

        def logged(word: str, freq: str | None = None, tag: str | None = None) -> None:
            row: dict[str, Any] = {"op": "add", "w": word, "f": None if freq is None else int(freq)}
            if tag:
                row["tag"] = tag
            row["via"] = "load_userdict"
            orig(word, freq, tag)
            row["freq"] = self.tok.FREQ[word]
            row.update(self.state([word]))
            self.emit(row)

        self.tok.add_word = logged
        try:
            self.tok.load_userdict(str(path))
        finally:
            del self.tok.add_word
        self.probe()


README_PROBES = [
    "如果放到post中将出错。",
    "「台中」正确应该不会被切开",
    "今天天气不错",
    "我来到北京清华大学",
    "他来到了网易杭研大厦",
]
USERDICT_SENT = (
    "李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
    "例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n"
    "「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。"
)
EXPLICIT_PROBES = [
    "李小福是创新办主任也是云计算方面的专家",
    "我用C++11和Python写easy_install脚本",
    "AT&T公司的e-mail地址是3.14%吗",
    "北京清华大学城的中华人民共和国中央人民政府",
    "abc ab a 的中的 😀笑 ＡＢＣ Edu Trust认证",
    "凱特琳是臺灣人，自定义词也行",
]
CUSTOM_PROBES = [
    "我爱北京天安门",
    "我来到北京清华大学",
    "中华人民共和国的人民",
    "我爱C++和AT&T的e-mail，3.14 hello abc ab",
    "ＡＢ😀臺灣凱特琳𠮷野家",
    "一一一了了的的",
    "中文英文",
    "",
    "，。 ",
    "首词第二前后空格千零零七全角阿拉伯空白多列负零",
]


def run_scenarios(sc: Scenarios, test_texts: Sequence[str]) -> None:
    # README: frequency tuning.
    sc.new("readme", README_PROBES)
    sc.sf(("中", "将"), True)
    sc.sf("台中", True)
    sc.add("台中")
    sc.sf(("今天", "天气"), True)
    sc.delete("今天天气")
    sc.delete("杭研")

    # test_userdict.py, on a fresh Tokenizer.
    sc.new("test_userdict", [USERDICT_SENT, "easy_install is great", "python 的正则表达式是好用的"])
    sc.userdict("userdict.txt")
    sc.add("石墨烯")
    sc.add("凱特琳")
    sc.delete("自定义词")
    for sent, seg in [
        ("今天天气不错", ("今天", "天气")),
        ("如果放到post中将出错。", ("中", "将")),
        ("我们中出了一个叛徒", ("中", "出")),
    ]:
        sc.probe([sent])
        sc.sf(seg, True, probe=False)
        sc.probe([sent])

    # add_word with explicit frequencies.
    sc.new("add_explicit", EXPLICIT_PROBES)
    for w, f in [
        ("创新办", 3),
        ("自定义词", 1),
        ("云计算", 10**9),
        ("easy_install", 3),
        ("C++11", 5),
        ("AT&T公司", 7),
        ("e-mail地址", 2),
        ("3.14%", 4),
        ("Edu Trust认证", 2000),
        ("北京清华大学", 50),
        ("清华大学城", 9),
        ("中华人民共和国中央人民政府", 1),
        ("ab", 2),
        ("abc", 3),
        ("Python", 100),
        ("的", 1),
        ("中", 0),
        ("凱特琳", 10**15),
        ("😀笑", 5),
        ("ＡＢＣ", 5),
        ("\u3000", 3),
        ("a", 1),
        ("", 5),
        ("", 0),
        ("创新办", 0),
        ("创新办", 8),
    ]:
        sc.add(w, f)

    # add_word without a frequency: suggest_freq(word) decides it.
    sc.new("add_suggested", EXPLICIT_PROBES[:4] + README_PROBES[:2])
    for w in [
        "石墨烯",
        "凱特琳",
        "台中",
        "创新办",
        "云计算",
        "easy_install",
        "C++",
        "AT&T",
        "中",
        "的",
        "清华大学",
        "北京清华大学",
        "abc",
        "Python3",
        "3.14%",
        "",
        " ",
        "😀",
        "中文😀",
        "\u9fd6\u9fd7",
        "𠮷野家",
        "中将",
        "石墨烯",
    ]:
        sc.add(w)

    # del_word, then adding the words back (Force_Split_Words keeps them).
    sc.new(
        "del_word",
        [*README_PROBES, "中国科学院计算技术研究所", "我爱北京天安门", "北京大厦了"],
    )
    for w in ["清华大学", "北京", "了", "中国科学院", "杭研", "大厦", "不存在的词"]:
        sc.delete(w)
    sc.add("清华大学", 5)
    sc.add("杭研")
    sc.add("北京", 10**6)

    # suggest_freq(..., tune=True) for words and for segments.
    sc.new(
        "suggest_tune",
        [
            "今天天气不错",
            "我们中出了一个叛徒",
            "如果放到post中将出错。",
            "「台中」正确应该不会被切开",
            "清华大学",
            "北京欢迎你",
            "ab abc",
        ],
    )
    for seg in [
        ("今天", "天气"),
        ("中", "出"),
        ("中", "将"),
        "台中",
        "韩玉赏鉴",
        ("清华", "大学"),
        ("北", "京"),
        ("我", "们"),
        ("a", "b"),
        ("",),
        (),
        ("中国", "科学院"),
        "中国科学院",
        ("中华", "人民", "共和国"),
        ("C", "++"),
        "清华大学",
        "",
    ]:
        sc.sf(seg, True)

    # suggest_freq without tuning, for many words and segment pairs.
    sc.new("suggest_many", [])
    import jieba

    queries: list[str | tuple[str, ...]] = []
    for t in test_texts:
        words = [w for w in jieba.lcut(t) if w.strip()]
        queries += [w for w in words if len(w) >= 2]
        queries += list(itertools.pairwise(words))
        queries.append(t)
    queries += ["", " ", "\u3000", "a b", "，", "😀", ("",), (), ("😀", "笑")]
    seen: set[Any] = set()
    for q in queries:
        if q not in seen:
            seen.add(q)
            sc.sf(q, False)

    # A negative frequency: `log` of it raises, on the routes it is on.
    sc.new("negative", ["中文", "我爱中文", "英文", "中", "，"])
    sc.add("中文", -5)
    sc.sf("中文", False)
    sc.sf(("中", "文"), False)
    sc.add("中文", 5)

    # Force_Split_Words is process-global: a new Tokenizer sees earlier del_word calls.
    sc.new("fs_carry_1", ["他来到了网易杭研大厦"])
    sc.delete("杭研")
    sc.new("fs_carry_2", ["他来到了网易杭研大厦"], keep_fs=True)

    # Custom dictionaries.
    for name in CUSTOM_DICTS:
        probes = [*CUSTOM_PROBES, *test_texts[:12]]
        sc.new(f"dict_{name}", probes, dictionary=name)
        for w, f in [("新词", None), ("中文", None), ("，", None), ("", None), ("中文", 3)]:
            sc.add(w, f, probe=False)
        sc.sf(("北京", "大学"), True, probe=False)
        sc.probe()

    # load_userdict on a custom dictionary.
    sc.new("dict_userdict", [USERDICT_SENT, *CUSTOM_PROBES[:4]], dictionary="basic.txt")
    sc.userdict("userdict.txt")


# --------------------------------------------------------------------------
# Recording everything


def read_texts(path: Path) -> list[Item]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        extra = {k: row[k] for k in ("k", "d") if k in row}
        items.append((row["t"], extra))
    return items


def write(path: Path, rows: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(r + "\n" for r in rows)
    log(f"wrote {path}: {len(rows)} rows, {path.stat().st_size / 1e3:.1f} kB")


def versions(jieba_only: bool) -> dict[str, str]:
    import importlib.metadata as md
    import platform
    import unicodedata

    import jieba

    if jieba.__version__ != JIEBA_VERSION:
        raise SystemExit(f"need jieba {JIEBA_VERSION}, found {jieba.__version__}")
    out = {
        "jieba": jieba.__version__,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "unicode": unicodedata.unidata_version,
    }
    if not jieba_only:
        out["datatrove"] = md.version("datatrove")
        out["spacy"] = md.version("spacy")
    return out


def record(args: argparse.Namespace) -> None:
    import logging

    import jieba

    jieba.setLogLevel(logging.WARNING)
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    env = versions(args.jieba_only)
    dt = None if args.jieba_only else Datatrove()
    jieba.dt.check_initialized()
    default_state = (len(jieba.dt.FREQ), jieba.dt.total)

    if args.texts_from:
        src = Path(args.texts_from)
        tests = read_texts(src / "golden-tests.jsonl")
        fineweb = read_texts(src / "golden-fineweb-1.jsonl")
        fineweb += read_texts(src / "golden-fineweb-2.jsonl")
        mixed = read_texts(src / "golden-mixed.jsonl")
    else:
        import random

        if not (args.jieba_src and args.fineweb):
            raise SystemExit("record needs --texts-from, or --jieba-src and --fineweb")
        tests = test_items(Path(args.jieba_src), Path(args.readme) if args.readme else None)
        paras = fineweb_paragraphs(Path(args.fineweb))
        rng = random.Random(SEED)
        fineweb = [(paras[i][0], {"d": paras[i][1]}) for i in pick(rng, len(paras), N_FINEWEB)]
        mixed = mixed_items([p for p, _ in paras], SEED + 1)
    fuzz = fuzz_items(SEED + 2, N_FUZZ)

    def every(i: int) -> bool:
        return True

    files: dict[str, list[str]] = {
        "golden-tests.jsonl": record_rows(tests, dt, every),
        "golden-fineweb-1.jsonl": record_rows(fineweb[: len(fineweb) // 2], dt, every),
        "golden-fineweb-2.jsonl": record_rows(fineweb[len(fineweb) // 2 :], dt, every),
        "golden-mixed.jsonl": record_rows(mixed, dt, every),
        "golden-fuzz.jsonl": record_rows(fuzz, dt, lambda i: i % FUZZ_DATATROVE_EVERY == 0),
    }
    if (len(jieba.dt.FREQ), jieba.dt.total) != default_state:
        raise AssertionError("the default dictionary changed while recording")
    test_texts = [t for t, _ in tests]
    with tempfile.TemporaryDirectory() as tmp:
        sc = Scenarios(Path(tmp))
        run_scenarios(sc, test_texts)
        files["golden-scenarios.jsonl"] = sc.rows
        if args.dict_small:
            sc.rows = []
            probes = test_texts + [t for t, _ in fineweb[:150]] + [t for t, _ in mixed[:100]]
            sc.new_small("dict_small", Path(args.dict_small), probes)
            for w in ["石墨烯", "台中", "凱特琳"]:
                sc.add(w, probe=False)
            sc.sf(("中", "将"), True, probe=False)
            sc.probe(README_PROBES)
            files["golden-dict-small.jsonl"] = sc.rows
    jieba.finalseg.Force_Split_Words.clear()

    meta: dict[str, Any] = {
        "recorded_with": env,
        "seed": SEED,
        "default_dict": {"len": default_state[0], "total": default_state[1]},
        "sources": {
            "tests": "jieba 0.42.1 sdist test/*.py and README.md code blocks (MIT)",
            "fineweb": "HuggingFaceFW/fineweb-2 cmn_Hani, 250 documents (ODC-By 1.0)",
            "userdict.txt": "jieba 0.42.1 test/userdict.txt (MIT), embedded in oracle.py, "
            f"sha256 {hashlib.sha256(USERDICT).hexdigest()}",
            "dict.txt.small": f"jieba 0.42.1 extra_dict/dict.txt.small, sha256 {DICT_SMALL_SHA256}",
        },
        "files": {},
    }
    for name, rows in files.items():
        write(out / name, rows)
        data = (out / name).read_bytes()
        meta["files"][name] = {"rows": len(rows), "sha256": hashlib.sha256(data).hexdigest()}
    (out / "recorded.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    total = sum((out / n).stat().st_size for n in files)
    log(f"{total / 1e3:.1f} kB of goldens, {env}")


# --------------------------------------------------------------------------
# Checks


def compare(args: argparse.Namespace) -> None:
    a, b = Path(args.a), Path(args.b)
    names = sorted({p.name for d in (a, b) for p in d.glob("golden-*.jsonl")})
    same = True
    for name in names:
        pa, pb = a / name, b / name
        if not (pa.exists() and pb.exists()):
            log(f"{name}: only in {'ab'[pb.exists()]}")
            same = False
            continue
        if pa.read_bytes() == pb.read_bytes():
            log(f"{name}: identical bytes")
            continue
        ra = pa.read_text(encoding="utf-8").splitlines()
        rb = pb.read_text(encoding="utf-8").splitlines()
        if args.jieba_only:

            def strip(rows: list[str]) -> list[dict[str, Any]]:
                out = []
                for r in rows:
                    row = json.loads(r)
                    for k in DATATROVE_FIELDS:
                        row.pop(k, None)
                    out.append(row)
                return out

            sa, sb = strip(ra), strip(rb)
            if sa == sb:
                log(f"{name}: identical without datatrove's fields")
                continue
            diffs = [i for i, (x, y) in enumerate(zip(sa, sb, strict=False)) if x != y]
        else:
            diffs = [i for i, (x, y) in enumerate(zip(ra, rb, strict=False)) if x != y]
        same = False
        log(f"{name}: {len(ra)} vs {len(rb)} rows, {len(diffs)} differ")
        for i in diffs[:5]:
            log(f"  line {i + 1}:\n    {ra[i][:300]}\n    {rb[i][:300]}")
    if not same:
        raise SystemExit(1)


def logcheck(args: argparse.Namespace) -> None:
    """`math.log(n) == math.log(float(n))` bit for bit for dict.txt's frequencies and total."""
    import platform

    if args.dict_txt:
        path = Path(args.dict_txt)
    else:
        import jieba

        path = Path(jieba.__file__).parent / "dict.txt"
    freqs: set[int] = set()
    total = 0
    for line in path.read_bytes().splitlines():
        _word, freq = line.strip().decode("utf-8").split(" ")[:2]
        freqs.add(int(freq))
        total += int(freq)
    values = sorted(n for n in freqs | {total} if n > 0)
    bad = [
        n for n in values if struct.pack("<d", math.log(n)) != struct.pack("<d", math.log(float(n)))
    ]
    digest = hashlib.sha256(b"".join(struct.pack("<d", math.log(n)) for n in values)).hexdigest()
    report = {
        "python": platform.python_version(),
        "values": len(values),
        "nonpositive": len(freqs) - len([n for n in freqs if n > 0]),
        "total": total,
        "mismatches": len(bad),
        "first_mismatches": bad[:10],
        "log_sha256": digest,
    }
    sys.stdout.write(json.dumps(report) + "\n")
    if bad:
        raise SystemExit(1)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="write the goldens and recorded.json into DIR")
    r.add_argument("dir")
    r.add_argument("--jieba-src", help="the unpacked jieba-0.42.1 sdist")
    r.add_argument("--readme", help="jieba's README.md")
    r.add_argument("--fineweb", help="FineWeb-2 cmn_Hani parquet")
    r.add_argument("--texts-from", help="take the corpora's texts from this recording")
    r.add_argument("--dict-small", help="jieba's extra_dict/dict.txt.small")
    r.add_argument("--jieba-only", action="store_true", help="leave out datatrove's fields")
    c = sub.add_parser("compare", help="compare two recordings")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--jieba-only", action="store_true", help="ignore datatrove's fields")
    lc = sub.add_parser("logcheck", help="math.log(int) against math.log(float)")
    lc.add_argument("dict_txt", nargs="?")
    args = p.parse_args(argv)
    {"record": record, "compare": compare, "logcheck": logcheck}[args.cmd](args)


if __name__ == "__main__":
    main()
