# Derived from jieba 0.42.1 (MIT), © 2013 Sun Junyi; modified.
"""Differential tests: pyroid.jieba must give jieba 0.42.1's results exactly.

Everything here runs against the live jieba (skipped if it is not installed):

- **Segmentation**: Hypothesis text that mixes dictionary words, CJK inside and outside
  ``re_han`` (U+4E00..U+9FD5), traditional characters, ASCII letters and digits, ``+#&._%-``,
  every kind of whitespace, ASCII and full-width punctuation and emoji, through ``lcut``
  (HMM on and off), ``lcut(cut_all=True)``, ``lcut_for_search``, ``tokenize``, ``cut`` and
  ``cut_for_search``, ``get_DAG``, ``calc``, ``suggest_freq``, bytes input and batches.
- **User dictionaries**: random sequences of ``add_word`` / ``del_word`` /
  ``suggest_freq(..., tune=True)`` / ``load_userdict`` on fresh tokenizers of both, then
  every mode, ``FREQ``, ``total`` and ``finalseg.Force_Split_Words``; ``load_userdict`` and
  custom dictionary files (``Tokenizer(dictionary=path)``) edge cases, error messages included
  (compared on the running CPython: ``math.log``'s message changed in 3.14).

``finalseg.Force_Split_Words`` is process-global in both libraries; the tests that change it
reset both and are ``thread_unsafe``. Lone surrogates and frequencies past 64 bits are
documented differences, tested at the end.
"""

# The test data is Chinese text and full-width punctuation on purpose.
# ruff: noqa: RUF001

from __future__ import annotations

import contextlib
import functools
import gc
import io
import logging
import os
import string
import tempfile
import types
import warnings
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

import pyroid.jieba as pj
from pyroid._pyroid import _testing
from pyroid._pyroid import jieba as pj_native

with warnings.catch_warnings():
    # jieba 0.42.1 has invalid escape sequences (SyntaxWarning when first compiled).
    warnings.simplefilter("ignore")
    jieba: Any = pytest.importorskip("jieba")
    finalseg: Any = pytest.importorskip("jieba.finalseg")
if jieba.__version__ != "0.42.1":
    pytest.skip(f"needs jieba 0.42.1, found {jieba.__version__}", allow_module_level=True)
jieba.setLogLevel(logging.WARNING)

examples = settings(suppress_health_check=[HealthCheck.too_slow])

# ---- reference state ---------------------------------------------------------------------


@functools.cache
def _jieba_template() -> Any:
    """A jieba Tokenizer with the default dictionary, never modified (copied per example)."""
    t = jieba.Tokenizer()
    t.check_initialized()
    return t


def fresh_jieba() -> Any:
    """A new jieba Tokenizer with the default dictionary loaded (a copy of the template)."""
    tmpl = _jieba_template()
    t = jieba.Tokenizer()
    t.FREQ = dict(tmpl.FREQ)
    t.total = tmpl.total
    t.initialized = True
    return t


def fresh_pyroid() -> pj.Tokenizer:
    t = pj.Tokenizer()
    t.initialize()
    return t


@functools.cache
def _dict_words() -> list[str]:
    return sorted(w for w, f in _jieba_template().FREQ.items() if f)


@contextlib.contextmanager
def clean_force_split() -> Iterator[None]:
    """Empty both libraries' ``Force_Split_Words`` before and after the block."""
    finalseg.Force_Split_Words.clear()
    pj_native.clear_force_split()
    try:
        yield
    finally:
        finalseg.Force_Split_Words.clear()
        pj_native.clear_force_split()


def outcome(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """``("ok", result)`` (generators listed) or ``("raise", type name, message)``."""
    try:
        r = fn(*args, **kwargs)
        if isinstance(r, types.GeneratorType):
            r = list(r)
        return ("ok", r)
    except Exception as e:
        return ("raise", type(e).__name__, str(e))


def assert_same(got: Any, want: Any, what: str) -> None:
    assert got == want, f"{what}\n pyroid: {got!r:.3000}\n  jieba: {want!r:.3000}"


# Every segmentation mode: ``f(module_or_tokenizer, x, **kw)``; ``kw`` is ``threads=`` for pyroid.
MODES: dict[str, Callable[..., Any]] = {
    "lcut": lambda m, x, **kw: m.lcut(x, **kw),
    "lcut_no_hmm": lambda m, x, **kw: m.lcut(x, HMM=False, **kw),
    "cut_all": lambda m, x, **kw: m.lcut(x, cut_all=True, **kw),
    "search": lambda m, x, **kw: m.lcut_for_search(x, **kw),
    "search_no_hmm": lambda m, x, **kw: m.lcut_for_search(x, HMM=False, **kw),
    "tokenize": lambda m, x, **kw: list(m.tokenize(x, **kw)),
    "tokenize_no_hmm": lambda m, x, **kw: list(m.tokenize(x, HMM=False, **kw)),
    "tokenize_search": lambda m, x, **kw: list(m.tokenize(x, mode="search", **kw)),
    "tokenize_search_no_hmm": lambda m, x, **kw: list(
        m.tokenize(x, mode="search", HMM=False, **kw)
    ),
}

# ---- strategies --------------------------------------------------------------------------

# jieba 0.42.1 test/test.py (a selection) and the README's examples.
JIEBA_TEST_SENTENCES = [
    "这是一个伸手不见五指的黑夜。我叫孙悟空，我爱北京，我爱Python和C++。",
    "我不喜欢日本和服。",
    "雷猴回归人间。",
    "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作",
    "我需要廉租房",
    "永和服装饰品有限公司",
    "我爱北京天安门",
    "abc",
    "隐马尔可夫",
    "雷猴是个好网站",
    "“Microsoft”一词由“MICROcomputer（微型计算机）”和“SOFTware（软件）”两部分组成",
    "草泥马和欺实马是今年的流行词汇",
    "伊藤洋华堂总府店",
    "中国科学院计算技术研究所",
    "罗密欧与朱丽叶",
    "我购买了道具和服装",
    "PS: 我觉得开源有一个好处，就是能够敦促自己不断改进，避免敞帚自珍",
    "湖北省石首市",
    "做好了这件事情就一了百了了",
    "的的的的的在的的的的就以和和和",
    "I love你，不以为耻，反以为rong",
    "因",
    "",
    "hello你好人们审美的观点是不同的",
    ",使用了它就可以解决一些问题",
    "干脆就把那部蒙人的闲法给废了拉倒！RT @laoshipukong : 27日，"
    "全国人大常委会第三次审议侵权责任法草案，删除了有关医疗损害责任“举证倒置”的规定。"
    "在医患纠纷中本已处于弱势地位的消费者由此将陷入万劫不复的境地。 ",
    "长春市长春节讲话",
    "结婚的和尚未结婚的",
    "BP神经网络如何训练才能在分类时增加区分度？",
    "南京市长江大桥",
    "应一些使用者的建议，也为了便于利用NiuTrans用于SMT研究",
    "程序员祝海林和朱会震是在孙健的左面和右面, 范凯在最右面.再往左是李松洪",
    "两块五一套，三块八一斤，四块七一本，五块六一条",
    "我是中华人民共和国公民;我爸爸是共和党党员; 地铁和平门站",
    "张晓梅去人民医院做了个B超然后去买了件T恤",
    "AT&T是一件不错的公司，给你发offer了吗？",
    "C++和c#是什么关系？11+122=133，是吗？PI=3.14159",
    "你认识那个和主席握手的的哥吗？他开一辆黑色的士。",
    "阿Q腰间挂着BB机手里拿着大哥大，说：我一般吃饭不AA制的。",
    "在1号店能买到小S和大S八卦的书，还有3D电视。",
    "看上去iphone8手机样式很赞,售价699美元,销量涨了5%么？",
    "我来到北京清华大学",
    "他来到了网易杭研大厦",
    "小明硕士毕业于中国科学院计算所，后在日本京都大学深造",
    "「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。",
]

HAN = st.characters(min_codepoint=0x4E00, max_codepoint=0x9FD5)
HAN_RUN = st.text(HAN, min_size=1, max_size=8)
# CJK that jieba's re_han does not cover: unified ideographs past U+9FD5, Extensions A and B,
# compatibility ideographs, radicals, and a few CJK symbols.
OTHER_HAN = st.one_of(
    st.characters(min_codepoint=0x9FD6, max_codepoint=0x9FFF),
    st.characters(min_codepoint=0x3400, max_codepoint=0x4DBF),
    st.characters(min_codepoint=0x20000, max_codepoint=0x2A6DF),
    st.characters(min_codepoint=0xF900, max_codepoint=0xFAFF),
    st.characters(min_codepoint=0x2E80, max_codepoint=0x2FDF),
    st.sampled_from(["〇", "々", "〆", "ㄅ", "ㄆ", "㐀", "\U00020000"]),
)
TRADITIONAL = st.sampled_from(
    [
        "凱特琳",
        "臺灣",
        "這個問題",
        "會議記錄",
        "學習資訊",
        "電腦網路",
        "開發環境",
        "「台中」正確應該不會被切開",
        "此時又可以分出來凱特琳了",
        "張晓梅",
    ]
)
ASCII_ALNUM = st.text(string.ascii_letters + string.digits, min_size=1, max_size=6)
SYMBOLS = st.text("+#&._%-", min_size=1, max_size=3)
NUMBERS = st.sampled_from(
    [
        "3.14",
        "3.14.15",
        "1.5e3",
        "5%",
        "12.5%",
        "100",
        "0.5",
        "1,000",
        "2024-01-01",
        "v1.2",
        "C++",
        "c#",
        "AT&T",
        "a.b_c-5%",
        "iphone8",
        "B超",
        "T恤",
        "x86_64",
        ".5",
        "5.",
        "%5",
        "-1",
        "3.5.6",
        "1.%",
    ]
)
# Whitespace to Python's \s (re_skip_default), and \r\n in both orders.
WHITESPACE = st.sampled_from(
    [
        "\r\n",
        "\n\r",
        "\r\r\n",
        "\r\n\n",
        "\u3000",
        "\xa0",
        "\u2028",
        "\u2029",
        "\x1c",
        "\x1d",
        "\x1e",
        "\x1f",
        "\x85",
        " ",
        "  ",
        "\t",
        "\n",
        "\r",
        "\x0b",
        "\x0c",
        "\u1680",
        "\u2000",
        "\u200a",
        "\u202f",
        "\u205f",
    ]
)
# Not whitespace to Python's \s.
NOT_WHITESPACE = st.sampled_from(["\u200b", "\u200d", "\ufeff", "\x00", "\x7f", "\u180e", "\x1b"])
PUNCT = st.sampled_from(
    [
        *"!\"'(),/:;<=>?@[\\]^`{|}~*$",
        *"，。！？；：、（）「」『』《》【】“”‘’…—～·．％＋＃＆－＿",
        "……",
        "——",
    ]
)
FOREIGN = st.sampled_from(
    [
        "ＡＢＣ",
        "１２３",
        "Ｐｙｔｈｏｎ",
        "ｶﾀｶﾅ",
        "한국어",
        "日本語のテキスト",
        "Ünïcödé",
        "ﬁ",
        "Ⅻ",
        "①②",
        "٣٤",
        "५",
        "e\u0301",
    ]
)
EMOJI = st.sampled_from(["😀", "👍🏽", "🇨🇳", "👨\u200d👩\u200d👧", "❤️", "☺", "🀄", "©", "™"])
ANY_CHAR = st.characters(exclude_categories=["Cs"])


def _dict_word() -> st.SearchStrategy[str]:
    return st.sampled_from(_dict_words())


DICT_WORD = st.deferred(_dict_word)
PIECE = st.one_of(
    DICT_WORD,
    DICT_WORD,
    DICT_WORD,
    HAN,
    HAN_RUN,
    OTHER_HAN,
    TRADITIONAL,
    ASCII_ALNUM,
    SYMBOLS,
    NUMBERS,
    WHITESPACE,
    NOT_WHITESPACE,
    PUNCT,
    FOREIGN,
    EMOJI,
    ANY_CHAR,
)
TEXT = st.one_of(
    st.lists(PIECE, max_size=24).map("".join),
    st.lists(PIECE, min_size=20, max_size=80).map("".join),
    st.sampled_from(JIEBA_TEST_SENTENCES),
    st.text(max_size=40),
)
EDGE_TEXTS = [
    "",
    " ",
    "\r\n",
    "\n\r",
    "中\r\n文",
    "a\r\r\nb",
    "北京\u3000大学",
    "\x1c中\x1d文\x1e\x1f\x85",
    "中\u200b文",
    "C++和c#",
    "3.14.15%",
    "AT&T",
    "iphone8手机",
    "的的的的的的的的的的",
    "杭研",
    "〇一二三",
    "鿕鿖",
    "\U00020000\U00020001中文",
    "a" * 300,
    "中" * 300,
    "B超和T恤",
    "5.5%的人",
    "e.g.和i.e.",
    "...---+++",
    "-1.5%",
    # Full mode buffers runs of ASCII letters and digits ("eng_buf") and can yield them late.
    "xAT&T",
    "C++x",
    "xB超",
    "abcAT&T3.14",
    "T恤abc",
    "a b超",
    "iPhone8是3.5%的",
]


# ---- segmentation --------------------------------------------------------------------------


@examples
@given(text=TEXT, hmm=st.booleans())
@example(text="我来到北京清华大学", hmm=True)
@example(text="他来到了网易杭研大厦", hmm=True)
@example(text="小明硕士毕业于中国科学院计算所，后在日本京都大学深造", hmm=False)
def test_lcut(text: str, hmm: bool) -> None:
    assert_same(pj.lcut(text, HMM=hmm), jieba.lcut(text, HMM=hmm), f"lcut({text!r}, HMM={hmm})")


@pytest.mark.parametrize("text", EDGE_TEXTS)
@pytest.mark.parametrize("mode", list(MODES))
def test_edge_texts(text: str, mode: str) -> None:
    f = MODES[mode]
    assert_same(f(pj, text), f(jieba, text), f"{mode} {text!r}")


@examples
@given(text=TEXT, hmm=st.booleans())
@example(text="我来到北京清华大学", hmm=True)
@example(text="a， b  c\r\n", hmm=True)
@example(text="C++和c#AT&T3.14", hmm=True)
def test_lcut_cut_all(text: str, hmm: bool) -> None:
    got = pj.lcut(text, cut_all=True, HMM=hmm)
    assert_same(got, jieba.lcut(text, cut_all=True, HMM=hmm), f"lcut({text!r}, cut_all=True)")


@examples
@given(text=TEXT, hmm=st.booleans())
@example(text="小明硕士毕业于中国科学院计算所，后在日本京都大学深造", hmm=True)
def test_lcut_for_search(text: str, hmm: bool) -> None:
    got = pj.lcut_for_search(text, HMM=hmm)
    assert_same(got, jieba.lcut_for_search(text, HMM=hmm), f"lcut_for_search({text!r}, {hmm})")


@examples
@given(text=TEXT, mode=st.sampled_from(["default", "search", "other"]), hmm=st.booleans())
@example(text="永和服装饰品有限公司", mode="search", hmm=True)
def test_tokenize(text: str, mode: str, hmm: bool) -> None:
    got = list(pj.tokenize(text, mode=mode, HMM=hmm))
    want = list(jieba.tokenize(text, mode=mode, HMM=hmm))
    assert_same(got, want, f"tokenize({text!r}, mode={mode!r}, HMM={hmm})")


@examples
@given(text=TEXT, cut_all=st.booleans(), hmm=st.booleans())
def test_cut_generators(text: str, cut_all: bool, hmm: bool) -> None:
    g = pj.cut(text, cut_all, hmm)
    assert isinstance(g, types.GeneratorType)
    assert_same(list(g), list(jieba.cut(text, cut_all, hmm)), f"cut({text!r}, {cut_all}, {hmm})")
    s = pj.cut_for_search(text, hmm)
    assert isinstance(s, types.GeneratorType)
    assert_same(list(s), list(jieba.cut_for_search(text, hmm)), f"cut_for_search({text!r})")


@examples
@given(text=TEXT)
@example(text="北京大学")
@example(text="AT&T 和 C++ 和 Edu Trust认证")
def test_get_dag_and_calc(text: str) -> None:
    dag = pj.get_DAG(text)
    want = jieba.get_DAG(text)
    assert_same(dag, want, f"get_DAG({text!r})")
    route: dict[int, Any] = {}
    ref_route: dict[int, Any] = {}
    pj.calc(text, want, route)
    jieba.calc(text, want, ref_route)
    assert_same(route, ref_route, f"calc({text!r})")


@examples
@given(word=st.one_of(TEXT, DICT_WORD), default=st.sampled_from([None, -1]))
def test_get_freq(word: str, default: int | None) -> None:
    pj.dt.check_initialized()
    jieba.dt.check_initialized()
    assert pj.get_FREQ(word, default) == jieba.get_FREQ(word, default)
    assert pj.get_FREQ(word) == jieba.get_FREQ(word)
    assert (word in pj.dt.FREQ) == (word in jieba.dt.FREQ)


SEGMENT = st.one_of(DICT_WORD, HAN, HAN_RUN, PIECE)


@examples
@given(
    segment=st.one_of(TEXT, st.lists(SEGMENT, max_size=4).map(tuple), st.lists(SEGMENT, max_size=3))
)
@example(segment=("今天", "天气"))
@example(segment=("中", "将"))
@example(segment=())
@example(segment="")
@example(segment="台中")
def test_suggest_freq(segment: str | tuple[str, ...] | list[str]) -> None:
    got = outcome(pj.suggest_freq, segment)
    want = outcome(jieba.suggest_freq, segment)
    assert_same(got, want, f"suggest_freq({segment!r})")


@examples
@given(
    docs=st.lists(TEXT, max_size=6),
    kind=st.sampled_from(["list", "tuple", "column", "large-column-2-chunks"]),
    threads=st.sampled_from([None, 1, 2]),
)
def test_batches(docs: list[str], kind: str, threads: int | None) -> None:
    x: Any
    if kind == "list":
        x = list(docs)
    elif kind == "tuple":
        x = tuple(docs)
    else:
        large = kind.startswith("large")
        x = _testing.column(docs, large=large, chunks=2 if large else 1)
    for name, f in MODES.items():
        got = f(pj, x, threads=threads)
        want = [f(jieba, d) for d in docs]
        assert_same(got, want, f"{name} over a {kind} of {docs!r}")


BYTES = st.one_of(
    TEXT.map(lambda s: s.encode("utf-8")),
    TEXT.map(lambda s: s.encode("gbk", "ignore")),
    TEXT.map(lambda s: s.encode("gb18030", "ignore")),
    st.binary(max_size=40),
)


@examples
@given(data=BYTES, as_bytearray=st.booleans())
@example(data="我来到北京清华大学".encode("gbk"), as_bytearray=False)
@example(data=b"\xff\xfe\x00abc", as_bytearray=False)
def test_bytes_input(data: bytes, as_bytearray: bool) -> None:
    x: bytes | bytearray = bytearray(data) if as_bytearray else data
    for name in ("lcut", "lcut_no_hmm", "cut_all", "search", "search_no_hmm"):
        f = MODES[name]
        assert_same(f(pj, x), f(jieba, x), f"{name}({x!r})")
    assert_same(list(pj.cut(x)), list(jieba.cut(x)), f"cut({x!r})")
    assert_same(list(pj.cut_for_search(x)), list(jieba.cut_for_search(x)), f"cut_for_search({x!r})")


@examples
@given(text=st.lists(st.one_of(TEXT, WHITESPACE), max_size=6).map("".join), cut_all=st.booleans())
@example(text="中\n\n文\r\n\x0c", cut_all=True)
def test_parallel_cut_is_jiebas(text: str, cut_all: bool) -> None:
    """jieba's ``_pcut`` (enable_parallel) maps ``lcut`` over ``splitlines(True)`` in a process
    pool; pyroid's runs the same pieces as one batch."""
    parts = text.splitlines(True)
    want = [w for p in parts for w in jieba.lcut(p, cut_all=cut_all)]
    assert_same(list(pj._pcut(text, cut_all=cut_all)), want, f"_pcut({text!r}, {cut_all})")
    want_search = [w for p in parts for w in jieba.lcut_for_search(p)]
    assert_same(list(pj._pcut_for_search(text)), want_search, f"_pcut_for_search({text!r})")


@pytest.mark.thread_unsafe(reason="1.1M code points per mode")
@pytest.mark.parametrize("mode", ["lcut", "cut_all", "search_no_hmm", "tokenize"])
def test_every_code_point(mode: str) -> None:
    """Each code point (lone surrogates aside) between two Chinese characters, in chunks: every
    character's class (re_han_default, Python's \\s, other) and how each mode treats it."""
    f = MODES[mode]
    cps = [cp for cp in range(0x110000) if not 0xD800 <= cp < 0xE000]
    for i in range(0, len(cps), 4096):
        text = "".join("中" + chr(cp) + "文" for cp in cps[i : i + 4096])
        got, want = f(pj, text), f(jieba, text)
        if got != want:
            bad = [
                cp
                for cp in cps[i : i + 4096]
                if f(pj, f"中{chr(cp)}文") != f(jieba, f"中{chr(cp)}文")
            ]
            pytest.fail(f"{mode}: {len(bad)} code points differ, e.g. {[hex(c) for c in bad[:20]]}")


@pytest.mark.parametrize("use_paddle", [True, False])
def test_use_paddle_without_paddle_is_ignored(use_paddle: bool) -> None:
    """use_paddle=True is ignored while paddle mode is not enabled, as in jieba."""
    text = JIEBA_TEST_SENTENCES[0]
    assert pj.lcut(text, use_paddle=use_paddle) == jieba.lcut(text, use_paddle=use_paddle)
    assert list(pj.cut(text, False, True, use_paddle)) == list(
        jieba.cut(text, False, True, use_paddle)
    )


def test_tokenize_rejects_bytes_like_jieba() -> None:
    got = outcome(pj.tokenize, "北京".encode())
    want = outcome(jieba.tokenize, "北京".encode())
    assert got == want
    assert got == ("raise", "ValueError", "jieba: the input parameter should be unicode.")


def test_default_freq_is_jiebas() -> None:
    """Every key and value of the default dictionary's FREQ, and its total."""
    t = fresh_pyroid()
    ref = _jieba_template()
    assert len(t.FREQ) == len(ref.FREQ) == 498_113
    assert t.total == ref.total == 60_101_967
    got = dict(t.FREQ.items())
    assert got.keys() == ref.FREQ.keys()
    diff = [(k, got[k], v) for k, v in ref.FREQ.items() if got[k] != v]
    assert not diff, diff[:20]


# ---- user dictionaries ------------------------------------------------------------------

SMALL_HAN = "我来到北京清华大学中国科学院计算所研究生命石墨烯云创新办主任台凱特琳的了是在"
OP_WORD = st.one_of(
    st.text(st.sampled_from(SMALL_HAN), min_size=1, max_size=5),
    DICT_WORD,
    st.tuples(DICT_WORD, DICT_WORD).map("".join),
    st.sampled_from(
        ["abc", "C++", "iphone8", "B超人", "Edu Trust认证", "a,b", "云计算", "台中", "", "中 文"]
    ),
    HAN_RUN,
)
FREQ_ARG = st.one_of(
    st.integers(0, 20),
    st.integers(0, 10**8),
    st.just(0),
    st.integers(-10, -1),
    st.sampled_from(["5", "+3", " 7 ", "1_000", "５", "٣"]),
    st.floats(0, 1e9, allow_nan=False),
)
TAG = st.one_of(st.none(), st.sampled_from(["n", "nz", "v", ""]))
USERDICT_LINE = st.builds(
    lambda w, f, t: " ".join(x for x in (w, f, t) if x is not None),
    st.text(st.sampled_from(SMALL_HAN), min_size=1, max_size=4),
    st.one_of(st.none(), st.integers(0, 50).map(str)),
    st.one_of(st.none(), st.sampled_from(["n", "nz", "N", "v1"])),
)
OPS = st.lists(
    st.one_of(
        st.tuples(st.just("add"), OP_WORD, FREQ_ARG, TAG),
        st.tuples(st.just("add_nofreq"), OP_WORD, TAG),
        st.tuples(st.just("del"), OP_WORD),
        st.tuples(st.just("tune_str"), OP_WORD),
        st.tuples(st.just("tune_tuple"), st.lists(OP_WORD, min_size=1, max_size=3)),
        st.tuples(st.just("suggest"), OP_WORD),
        st.tuples(st.just("load"), st.lists(USERDICT_LINE, max_size=4)),
        st.tuples(st.just("add_bytes"), OP_WORD, FREQ_ARG),
    ),
    max_size=8,
)
EXTRA_TEXT = st.lists(
    st.one_of(st.text(st.sampled_from(SMALL_HAN), max_size=6), PIECE), max_size=10
).map("".join)


def apply_op(t: Any, op: tuple[Any, ...]) -> tuple[Any, ...]:
    kind = op[0]
    if kind == "add":
        return outcome(t.add_word, op[1], op[2], op[3])
    if kind == "add_nofreq":
        return outcome(t.add_word, op[1], tag=op[2])
    if kind == "del":
        return outcome(t.del_word, op[1])
    if kind == "tune_str":
        return outcome(t.suggest_freq, op[1], True)
    if kind == "tune_tuple":
        return outcome(t.suggest_freq, tuple(op[1]), True)
    if kind == "suggest":
        return outcome(t.suggest_freq, op[1])
    if kind == "load":
        return outcome(t.load_userdict, io.BytesIO("\n".join(op[1]).encode()))
    if kind == "add_bytes":
        return outcome(t.add_word, op[1].encode("utf-8"), op[2])
    raise AssertionError(kind)


def op_words(ops: Sequence[tuple[Any, ...]]) -> list[str]:
    words: list[str] = []
    for op in ops:
        if op[0] == "tune_tuple":
            words += [*op[1], "".join(op[1])]
        elif op[0] == "load":
            words += [ln.split(" ")[0] for ln in op[1]]
        else:
            words.append(op[1])
    return words


def compare_tokenizers(p: Any, j: Any, texts: Sequence[str], words: Sequence[str]) -> None:
    """Every mode on ``texts``; FREQ of ``words`` and their prefixes; total, len, tags."""
    assert p.total == j.total
    assert len(p.FREQ) == len(j.FREQ)
    for w in words:
        for k in [w[: i + 1] for i in range(len(w))] or [w]:
            assert p.FREQ.get(k) == j.FREQ.get(k), k
            assert (k in p.FREQ) == (k in j.FREQ), k
    assert p.user_word_tag_tab == j.user_word_tag_tab
    assert pj_native.force_split_words() == sorted(finalseg.Force_Split_Words)
    for text in texts:
        for name, f in MODES.items():
            assert_same(outcome(f, p, text), outcome(f, j, text), f"{name} {text!r}")
        assert_same(outcome(p.get_DAG, text), outcome(j.get_DAG, text), f"get_DAG {text!r}")


@pytest.mark.thread_unsafe(reason="finalseg.Force_Split_Words is process-global")
@examples
@given(ops=OPS, extra=EXTRA_TEXT)
@example(
    ops=[("add_nofreq", "石墨烯", None), ("add_nofreq", "凱特琳", None), ("del", "自定义词")],
    extra="石墨烯和凱特琳的自定义词",
)
@example(ops=[("del", "北京"), ("del", "大学")], extra="我来到北京清华大学")
@example(ops=[("tune_tuple", ["中", "将"]), ("tune_str", "台中")], extra="「台中」")
@example(ops=[("add", "北京大学", -3, None)], extra="北京大学")
def test_user_dictionary_ops(ops: list[tuple[Any, ...]], extra: str) -> None:
    words = op_words(ops)
    texts = [
        "".join(words),
        "，".join(words),
        " ".join(words),
        extra,
        "如果放到post中将出错。",
        "我们中出了一个叛徒",
        *JIEBA_TEST_SENTENCES[:2],
    ]
    with clean_force_split():
        p, j = fresh_pyroid(), fresh_jieba()
        for op in ops:
            assert_same(apply_op(p, op), apply_op(j, op), f"op {op!r}")
        compare_tokenizers(p, j, texts, words)


@pytest.mark.thread_unsafe(reason="finalseg.Force_Split_Words is process-global")
def test_total_zero_after_add_word() -> None:
    """A negative add_word that brings total to 0: cutting and suggest_freq raise as in jieba."""
    with clean_force_split():
        p, j = fresh_pyroid(), fresh_jieba()
        for t in (p, j):
            t.add_word("某某", -t.total)
        assert p.total == j.total == 0
        segments: list[Any] = ["北京", "，", " 北京", "，北京", "", ("北", "京"), (), ("，",)]
        for seg in segments:
            assert_same(outcome(p.suggest_freq, seg), outcome(j.suggest_freq, seg), repr(seg))
        assert_same(outcome(p.add_word, "新词"), outcome(j.add_word, "新词"), "add_word")
        compare_tokenizers(p, j, ["北京", "，", "abc", "", " 。"], ["某某"])


@pytest.mark.thread_unsafe(reason="finalseg.Force_Split_Words is process-global")
def test_del_word_force_split_reaches_the_hmm() -> None:
    """del_word adds the word to Force_Split_Words: the HMM may no longer produce it."""
    text = "他来到了网易杭研大厦"
    with clean_force_split():
        p, j = fresh_pyroid(), fresh_jieba()
        assert "杭研" in p.lcut(text)
        for t in (p, j):
            t.del_word("杭研")
        assert_same(p.lcut(text), j.lcut(text), "after del_word")
        assert "杭研" not in p.lcut(text)
        # Force_Split_Words is shared by every tokenizer, in both libraries.
        assert_same(fresh_pyroid().lcut(text), fresh_jieba().lcut(text), "another tokenizer")


# jieba 0.42.1 test/userdict.txt (with its BOM and its last line of spaces).
SAMPLE_USERDICT = (
    "\ufeff云计算 5\n李小福 2 nr\n创新办 3 i\neasy_install 3 eng\n好用 300\n韩玉赏鉴 3 nz\n"
    "八一双鹿 3 nz\n台中\n凱特琳 nz\nEdu Trust认证 2000\n      "
).encode()

USERDICT_CASES: dict[str, bytes] = {
    "sample": SAMPLE_USERDICT,
    "bom-only": b"\xef\xbb\xbf\n",
    "bom-word": "\ufeff石墨烯 7 n\n".encode(),
    "blank-lines": "\n\n  \n\t\n云计算 5\n\n".encode(),
    "word-with-spaces": "Edu Trust认证 2000\na b c\n中 文 3 n\n".encode(),
    "tag-without-freq": "凱特琳 nz\n台中 n\n".encode(),
    "freq-without-tag": "云计算 5\n创新办 0\n".encode(),
    "freq-then-letters": "词语 12abc\n".encode(),
    "uppercase-tag": "词 3 N\n".encode(),
    "digits-in-tag": "词 3 n1\n".encode(),
    "leading-zeros": "石墨烯 007\n".encode(),
    "tab-separated": "石墨烯\t5\n".encode(),
    "extra-field": "北京 5 n extra\n".encode(),
    "crlf": "云计算 5 n\r\n创新办 3 i\r\n台中\r\n".encode(),
    "cr-only": "云计算 5\r创新办 3\r".encode(),
    "number-word": b"123\n12 5\n",
    "surrounding-spaces": "  石墨烯 5  \n".encode(),
    "vertical-tab": "\x0b石墨烯 5\x0b\n\x0c凱特琳 3\x0c\n".encode(),
    "unicode-spaces": "\u3000石墨烯 5\u3000\n\x85凱特琳 3\n".encode(),
    "nbsp": "石墨烯\xa05\n".encode(),
    "zero-freq": "北京 0\n大学 0\n".encode(),
    "empty": b"",
    "no-final-newline": "云计算 5".encode(),
    "duplicates": "云计算 5\n云计算 9\n".encode(),
    "not-utf8": b"\xff\xfe\n",
    "gbk": "云计算 5\n".encode("gbk"),
    "valid-then-invalid": "云计算 5\n".encode() + b"\xe4\xb8\n" + "石墨烯 3\n".encode(),
}
USERDICT_TEXTS = [
    "李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿",
    "「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。",
    "easy_install is great",
    "北京大学的Edu Trust认证",
]


def _userdict_words(content: bytes) -> list[str]:
    text = content.decode("utf-8", "replace").lstrip("\ufeff")
    return [w for ln in text.splitlines() for w in ln.strip().split(" ")[:1] if w]


@contextlib.contextmanager
def unclosed_files_ignored() -> Iterator[None]:
    """jieba leaves files open: ``load_userdict(path)`` always, and the dictionary file when
    it fails to parse. The ResourceWarning that follows is not what is compared here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        yield
        gc.collect()


def load_userdict_path(t: Any, path: str) -> tuple[Any, ...]:
    with unclosed_files_ignored():
        return outcome(t.load_userdict, path)


@pytest.mark.thread_unsafe(reason="finalseg.Force_Split_Words is process-global; temp files")
@pytest.mark.parametrize("source", ["path", "binary-file", "text-file", "bytesio", "stringio"])
@pytest.mark.parametrize("case", list(USERDICT_CASES))
def test_load_userdict(case: str, source: str, tmp_path: Path) -> None:
    content = USERDICT_CASES[case]
    path = tmp_path / "userdict.txt"
    path.write_bytes(content)
    shared = io.BytesIO(content)  # one object for both, so repr(f) in messages is the same

    def load(t: Any) -> tuple[Any, ...]:
        if source == "path":
            return load_userdict_path(t, str(path))
        if source == "bytesio":
            shared.seek(0)
            return outcome(t.load_userdict, shared)
        if source == "stringio":
            return outcome(t.load_userdict, io.StringIO(content.decode("utf-8", "replace")))
        if source == "binary-file":
            with open(path, "rb") as fb:
                return outcome(t.load_userdict, fb)
        with open(path, encoding="utf-8") as ft:
            return outcome(t.load_userdict, ft)

    with clean_force_split():
        p, j = fresh_pyroid(), fresh_jieba()
        assert_same(load(p), load(j), f"load_userdict({case}, {source})")
        compare_tokenizers(p, j, USERDICT_TEXTS, _userdict_words(content))


@pytest.mark.thread_unsafe(reason="finalseg.Force_Split_Words is process-global; temp files")
def test_jieba_test_userdict_scenario(tmp_path: Path) -> None:
    """jieba 0.42.1 test/test_userdict.py on fresh tokenizers (its posseg part aside)."""
    path = tmp_path / "userdict.txt"
    path.write_bytes(SAMPLE_USERDICT)
    test_sent = (
        "李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
        "例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n"
        "「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。"
    )
    results = []
    with clean_force_split():
        for t in (pj.Tokenizer(), fresh_jieba()):
            assert load_userdict_path(t, str(path)) == ("ok", None)
            t.add_word("石墨烯")
            t.add_word("凱特琳")
            t.del_word("自定义词")
            r: list[Any] = [
                "/".join(t.cut(test_sent)),
                "/".join(t.cut("easy_install is great")),
                "/".join(t.cut("python 的正则表达式是好用的")),
            ]
            for sent, seg in [
                ("今天天气不错", ("今天", "天气")),
                ("如果放到post中将出错。", ("中", "将")),
                ("我们中出了一个叛徒", ("中", "出")),
            ]:
                word = "".join(seg)
                r.append("/".join(t.cut(sent, HMM=False)))
                r.append((t.FREQ.get(word), t.suggest_freq(seg, True)))
                r.append("/".join(t.cut(sent, HMM=False)))
            r.append(sorted(t.user_word_tag_tab.items()))
            results.append(r)
    assert_same(results[0], results[1], "test_userdict.py")
    assert "石墨烯" in results[0][0].split("/")
    assert "凱特琳" in results[0][0].split("/")


# ---- custom dictionaries ------------------------------------------------------------------

CUSTOM_DICTS: dict[str, bytes] = {
    "small": "北京 5 ns\n大学 3 n\n北京大学 2 nt\n清华 4\n".encode(),
    "duplicates": "北京 5\n北京 7\n大学 3\n大学 1\n".encode(),
    "freq-zero": "北京 0\n大学 3\n北京大学 0\n".encode(),
    "prefix-then-word": "北 3\n北京 5\n北京大学 2\n".encode(),
    "word-then-prefix": "北京大学 2\n北京 5\n北 3\n".encode(),
    "non-cjk": b"hello 5\nC++ 3\nAT&T 2\n3.14 1\nEdu 4\n",
    "plus-sign": "北京 +5\n".encode(),
    "tab-before-freq": "北京 \t5\n".encode(),
    "underscore": "北京 1_000\n大学 2\n".encode(),
    "fullwidth-digit": "北京 ５\n大学 ٣\n".encode(),
    "negative": "北京 -5\n大学 10\n".encode(),
    "negative-total": "北京 -50\n大学 10\n".encode(),
    "zero-total": "北京 5\n大学 -5\n".encode(),
    "negative-prefix": "北 -1\n北京 5\n".encode(),
    "tagged": "北京 5 ns extra fields\n".encode(),
    "bom": "\ufeff北京 5\n大学 3\n".encode(),
    "crlf": "北京 5\r\n大学 3\r\n".encode(),
    "no-final-newline": "北京 5\n大学 3".encode(),
    "vertical-tab": "\x0b北京 5\n大学 3\x0b\n".encode(),
    "form-feed": "\x0c北京 5\n".encode(),
    "nbsp": "\xa0北京 5\n".encode(),
    "ideographic-space": "\u3000北京 5\n".encode(),
    "empty": b"",
    "only-newline": b"\n",
    "invalid-no-freq": "北京\n".encode(),
    "invalid-freq": "北京 abc\n".encode(),
    "invalid-float": "北京 5.0\n".encode(),
    "invalid-two-spaces": "北京  5\n".encode(),
    "invalid-blank-line": "北京 5\n\n大学 3\n".encode(),
    "invalid-utf8": b"\xff 5\n",
    "invalid-later-line": "北京 5\n大学 3\n清华\n".encode(),
}
CUSTOM_TEXTS = [
    "北京大学",
    "北京",
    "大学",
    "清华北京",
    "hello C++ AT&T 3.14",
    "，",
    "",
    "北京，大学。北",
]


def _custom_dict_outcomes(t: Any, texts: Sequence[str], *, suggest: bool = True) -> list[Any]:
    with unclosed_files_ignored():
        out: list[Any] = [outcome(t.initialize)]
        if out[0][0] == "ok":
            out += [t.total, len(t.FREQ), sorted(t.FREQ.items())]
        for text in texts:
            out += [(name, text, outcome(f, t, text)) for name, f in MODES.items()]
            out.append(("get_DAG", text, outcome(t.get_DAG, text)))
        if suggest:
            segments: list[Any] = ["北京", "，", ("北", "京"), ()]
            out += [("suggest", s, outcome(t.suggest_freq, s)) for s in segments]
    return out


def _jieba_custom(path: str, cache_dir: str) -> Any:
    t = jieba.Tokenizer(path)
    t.tmp_dir = cache_dir  # its cache file goes there, not to the shared temp dir
    return t


@pytest.mark.thread_unsafe(reason="catch_warnings (jieba leaves files open)")
@pytest.mark.parametrize("case", list(CUSTOM_DICTS))
def test_custom_dictionary(case: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dict.txt")
        Path(path).write_bytes(CUSTOM_DICTS[case])
        got = _custom_dict_outcomes(pj.Tokenizer(path), CUSTOM_TEXTS)
        want = _custom_dict_outcomes(_jieba_custom(path, d), CUSTOM_TEXTS)
        assert_same(got, want, f"Tokenizer(dictionary={case})")


NO_WORD_BLOCK = ["", "，", "。 ！\r\n", "\u3000\u3000", "😀"]


@pytest.mark.thread_unsafe(reason="catch_warnings (jieba leaves files open)")
@pytest.mark.parametrize("dictionary", ["default", "missing", "invalid"])
def test_dictionary_loads_on_the_first_word_block(dictionary: str) -> None:
    """jieba loads the dictionary in ``get_DAG``, on the first run of ``re_han_default``
    characters: text without one is cut without loading it (or failing to load it)."""
    with tempfile.TemporaryDirectory() as d:
        path: str | None = None
        if dictionary != "default":
            path = os.path.join(d, "dict.txt")
            if dictionary == "invalid":
                Path(path).write_bytes("北京\n".encode())
        for text in NO_WORD_BLOCK:
            for name, f in MODES.items():
                p = pj.Tokenizer(path)
                j = jieba.Tokenizer(path)
                j.tmp_dir = d
                with unclosed_files_ignored():
                    got = (outcome(f, p, text), p.initialized, len(p.FREQ))
                    want = (outcome(f, j, text), j.initialized, len(j.FREQ))
                assert_same(got, want, f"{name}({text!r}) with the {dictionary} dictionary")


@pytest.mark.thread_unsafe(reason="catch_warnings (jieba leaves files open)")
def test_custom_dictionary_with_a_63_bit_frequency() -> None:
    """2**63 - 1 loads and cuts as in jieba; suggest_freq's answer, 2**63 + 1, does not fit
    in 64 bits, so pyroid raises OverflowError there (documented)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dict.txt")
        Path(path).write_bytes("北京 9223372036854775807\n".encode())
        p, j = pj.Tokenizer(path), _jieba_custom(path, d)
        got = _custom_dict_outcomes(p, CUSTOM_TEXTS, suggest=False)
        want = _custom_dict_outcomes(j, CUSTOM_TEXTS, suggest=False)
        assert_same(got, want, "Tokenizer(dictionary=big-freq)")
        assert j.suggest_freq("北京") == 2**63 + 1
        with pytest.raises(OverflowError):
            p.suggest_freq("北京")


DICT_LINE_WORD = st.one_of(
    st.text(st.sampled_from("北京大学清华中国人民共和"), min_size=1, max_size=4),
    st.sampled_from(["hello", "C++", "AT&T", "3.14", "a"]),
)
DICT_FREQ = st.one_of(
    st.integers(0, 1000).map(str),
    st.sampled_from(["0", "+5", "-3", "1_0", "５", "\t5", "5.0", "", "x"]),
)
DICT_LINE = st.one_of(
    st.builds(
        lambda w, f, t: " ".join(x for x in (w, f, t) if x is not None),
        DICT_LINE_WORD,
        DICT_FREQ,
        st.one_of(st.none(), st.sampled_from(["n", "ns", "x y"])),
    ),
    st.sampled_from(["", " ", "北京", "北京  5"]),
)
STRIPPABLE = st.sampled_from(
    ["", "", "", " ", "\t", "\r", "\x0b", "\x0c", "\x1c", "\x85", "\ufeff"]
)


@pytest.mark.thread_unsafe(reason="catch_warnings (jieba leaves files open)")
@examples
@given(
    lines=st.lists(st.tuples(STRIPPABLE, DICT_LINE, STRIPPABLE), max_size=6),
    eol=st.sampled_from(["\n", "\r\n"]),
    final_eol=st.booleans(),
    texts=st.lists(st.text(st.sampled_from("北京大学清华中国人民共和，"), max_size=8), max_size=3),
)
@example(lines=[("\x0b", "北京 5", "")], eol="\n", final_eol=True, texts=["北京"])
def test_custom_dictionary_files(
    lines: list[tuple[str, str, str]], eol: str, final_eol: bool, texts: list[str]
) -> None:
    content = eol.join(a + ln + b for a, ln, b in lines) + (eol if final_eol and lines else "")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dict.txt")
        Path(path).write_bytes(content.encode())
        got = _custom_dict_outcomes(pj.Tokenizer(path), texts)
        want = _custom_dict_outcomes(_jieba_custom(path, d), texts)
        assert_same(got, want, f"dictionary file {content!r}")


@pytest.mark.parametrize("case", ["small", "duplicates", "negative", "bom", "invalid-utf8"])
def test_gen_pfdict(case: str) -> None:
    def named(content: bytes) -> io.BytesIO:
        f = io.BytesIO(content)
        f.name = "dict.txt"  # the error message names the file
        return f

    content = CUSTOM_DICTS[case]
    got = outcome(pj.Tokenizer.gen_pfdict, named(content))
    want = outcome(jieba.Tokenizer.gen_pfdict, named(content))
    assert_same(got, want, f"gen_pfdict({case})")


@pytest.mark.thread_unsafe(reason="temp files")
def test_set_dictionary_and_initialize(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_bytes(CUSTOM_DICTS["small"])
    b.write_bytes(CUSTOM_DICTS["prefix-then-word"])
    missing = str(tmp_path / "missing.txt")
    text = "北京大学的清华"
    results = []
    for t in (pj.Tokenizer(), fresh_jieba()):
        t.tmp_dir = str(tmp_path)
        r: list[Any] = [repr(t), t.lcut(text)]
        t.add_word("的清华", 9)
        r.append(t.lcut(text))
        t.set_dictionary(str(a))
        r += [repr(t), t.dictionary, t.initialized, t.lcut(text), t.total, len(t.FREQ)]
        t.initialize(str(b))
        r += [t.dictionary, t.lcut(text), t.total, len(t.FREQ)]
        t.initialize(str(b))  # the same file again: nothing to do
        r += [t.initialized, t.total]
        r.append(outcome(t.set_dictionary, missing))
        r += [t.dictionary, t.lcut(text)]
        results.append(r)
    assert_same(results[0], results[1], "set_dictionary / initialize")
    assert repr(pj.Tokenizer("dict.txt")) == repr(jieba.Tokenizer("dict.txt"))
    assert pj.Tokenizer("x/../dict.txt").dictionary == jieba.Tokenizer("x/../dict.txt").dictionary


# ---- documented differences -----------------------------------------------------------------

SURROGATE = st.characters(categories=["Cs"])
SURROGATE_CALLS: dict[str, Callable[[str], Any]] = {
    "lcut": lambda x: pj.lcut(x),
    "lcut_no_hmm": lambda x: pj.lcut(x, HMM=False),
    "cut_all": lambda x: pj.lcut(x, cut_all=True),
    "search": lambda x: pj.lcut_for_search(x),
    "tokenize": lambda x: list(pj.tokenize(x)),
    "cut": lambda x: list(pj.cut(x)),
    "get_DAG": lambda x: pj.get_DAG(x),
    "batch": lambda x: pj.lcut(["ok", x]),
}


@examples
@given(
    text=st.tuples(TEXT, SURROGATE, TEXT).map("".join), call=st.sampled_from(list(SURROGATE_CALLS))
)
def test_lone_surrogates_raise(text: str, call: str) -> None:
    """jieba treats a lone surrogate as a character; pyroid raises UnicodeEncodeError."""
    jieba.lcut(text)
    with pytest.raises(UnicodeEncodeError):
        SURROGATE_CALLS[call](text)


@pytest.mark.parametrize("big", [2**63, -(2**63) - 1, 10**30])
def test_frequencies_beyond_64_bits_raise(big: int) -> None:
    """jieba takes any int as a frequency; pyroid raises OverflowError (documented)."""
    j = fresh_jieba()
    j.add_word("大数", big)
    assert j.FREQ["大数"] == big
    with pytest.raises(OverflowError):
        fresh_pyroid().add_word("大数", big)
