# Derived from jieba 0.42.1 (MIT), © 2013 Sun Junyi; modified.
# The test data is Chinese text and full-width punctuation on purpose.
# ruff: noqa: RUF001
"""pyroid.jieba: pyroid's conventions, the FREQ mapping, parallel mode and the delegated parts.

- **Containers**: str in, list of words out; a list, tuple or Arrow string column (``string``,
  ``large_string``, ``string_view``, sliced, several chunks, or ``_testing.column`` without
  pyarrow) in, one list per document out (``None`` for an Arrow null), for ``lcut``,
  ``lcut_for_search``, ``tokenize``, ``cut`` and ``cut_for_search``.
- **Threads**: identical results for ``threads=None/1/2/4``; a 10k-document batch under a timeout;
  batches large enough that their words are interned on the pool in several tasks, alone and
  from several threads with ``gc.collect()`` running.
- **Errors**: ``InputError`` for non-str items and wrong input types, ``ValueError`` /
  ``InputError`` for bad ``threads``, ``UnicodeEncodeError`` with a note for lone surrogates; a
  failing batch raises its lowest failing document's error, with a note giving its index.
- **FREQ**: a read-only mapping, empty until the dictionary loads, then jieba's 498,113 entries.
- **Delegated**: ``pyroid.jieba.posseg`` and ``pyroid.jieba.analyse`` give jieba's results,
  see the words added through ``pyroid.jieba``, and warn ``pyroid.FallbackWarning`` once per
  feature; ``PyroidError`` when jieba is missing. These run in child interpreters, since they
  change ``jieba.dt`` and the once-per-process warning state.

The comparisons with jieba itself are in test_jieba_parity.py.
"""

from __future__ import annotations

import collections.abc
import json
import os
import subprocess
import sysconfig
import tempfile
import types
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import pyroid
import pyroid.jieba as pj
from pyroid._pyroid import _testing

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
RunPython = Callable[..., subprocess.CompletedProcess[str]]

DOCS = [
    "我来到北京清华大学",
    "他来到了网易杭研大厦",
    "",
    "小明硕士毕业于中国科学院计算所，后在日本京都大学深造",
    "Python和C++，AT&T的3.14%",
    " \r\n\u3000",
    "永和服装饰品有限公司",
    "「台中」正確應該不會被切開。",
]

# Every mode: (name, call on a str or a batch).
MODES: dict[str, Callable[..., Any]] = {
    "lcut": lambda x, **kw: pj.lcut(x, **kw),
    "lcut_no_hmm": lambda x, **kw: pj.lcut(x, HMM=False, **kw),
    "cut_all": lambda x, **kw: pj.lcut(x, cut_all=True, **kw),
    "search": lambda x, **kw: pj.lcut_for_search(x, **kw),
    "search_no_hmm": lambda x, **kw: pj.lcut_for_search(x, HMM=False, **kw),
    "tokenize": lambda x, **kw: list(pj.tokenize(x, **kw)),
    "tokenize_search": lambda x, **kw: list(pj.tokenize(x, mode="search", **kw)),
    "cut": lambda x, **kw: list(pj.cut(x, **kw)),
    "cut_for_search": lambda x, **kw: list(pj.cut_for_search(x, **kw)),
}


def per_doc(mode: str, docs: list[str | None]) -> list[Any]:
    return [None if d is None else MODES[mode](d) for d in docs]


# ---- jieba's README examples -------------------------------------------------------------


def test_readme_examples() -> None:
    assert pj.lcut("我来到北京清华大学") == ["我", "来到", "北京", "清华大学"]
    assert pj.lcut("我来到北京清华大学", cut_all=True) == [
        "我",
        "来到",
        "北京",
        "清华",
        "清华大学",
        "华大",
        "大学",
    ]
    assert pj.lcut("他来到了网易杭研大厦") == ["他", "来到", "了", "网易", "杭研", "大厦"]
    assert pj.lcut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造") == [
        *("小明", "硕士", "毕业", "于", "中国", "科学", "学院", "科学院", "中国科学院"),
        *("计算", "计算所", "，", "后", "在", "日本", "京都", "大学", "日本京都大学", "深造"),
    ]
    assert list(pj.tokenize("永和服装饰品有限公司")) == [
        ("永和", 0, 2),
        ("服装", 2, 4),
        ("饰品", 4, 6),
        ("有限公司", 6, 10),
    ]
    assert list(pj.tokenize("永和服装饰品有限公司", mode="search")) == [
        ("永和", 0, 2),
        ("服装", 2, 4),
        ("饰品", 4, 6),
        ("有限", 6, 8),
        ("公司", 8, 10),
        ("有限公司", 6, 10),
    ]


def test_generators() -> None:
    t = pj.Tokenizer()
    for g in (
        pj.cut(DOCS[0]),
        pj.cut(DOCS[0], cut_all=True),
        pj.cut_for_search(DOCS[0]),
        pj.tokenize(DOCS[0]),
        pj.cut(DOCS),
        pj.tokenize(DOCS, mode="search"),
        t.cut(DOCS[0]),
        t.cut_for_search(DOCS[0]),
        t.tokenize(DOCS[0]),
    ):
        assert isinstance(g, types.GeneratorType)
    assert list(pj.cut(DOCS[0])) == pj.lcut(DOCS[0])
    assert list(pj.cut(DOCS)) == pj.lcut(DOCS)
    assert list(pj.cut_for_search(DOCS)) == pj.lcut_for_search(DOCS)


def test_empty() -> None:
    for mode, f in MODES.items():
        assert f("") == [], mode
        assert f([]) == [], mode
        assert f(()) == [], mode
        assert f([""]) == [[]], mode
        assert f(_testing.column([])) == [], mode


# ---- containers ----------------------------------------------------------------------------


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("kind", ["list", "tuple"])
def test_list_and_tuple(kind: str, mode: str) -> None:
    x = list(DOCS) if kind == "list" else tuple(DOCS)
    out = MODES[mode](x)
    assert type(out) is list
    assert all(type(r) is list for r in out)
    assert out == per_doc(mode, list(DOCS))
    if mode.startswith("tokenize"):
        assert all(type(tok) is tuple for r in out for tok in r)


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("chunks", [1, 3])
@pytest.mark.parametrize("large", [False, True], ids=["string", "large_string"])
def test_column_without_pyarrow(large: bool, chunks: int, mode: str) -> None:
    values: list[str | None] = [*DOCS[:3], None, *DOCS[3:], None]
    col = _testing.column(values, large=large, chunks=chunks, name="text")
    assert MODES[mode](col) == per_doc(mode, values)


ARROW_TYPES = ["string", "large_string", "string_view"]


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("type_name", ARROW_TYPES)
def test_pyarrow(pa: Any, type_name: str, mode: str) -> None:
    values: list[str | None] = [None, *DOCS, None]
    ty = getattr(pa, type_name)()
    arr = pa.array(values, ty)
    f = MODES[mode]
    assert f(arr) == per_doc(mode, values)
    # Sliced (offset > 0), and several chunks.
    assert f(arr[2:7]) == per_doc(mode, values[2:7])
    chunked = pa.chunked_array([pa.array(values[:4], ty), pa.array([], ty), arr[4:]])
    assert f(chunked) == per_doc(mode, values)
    assert f(pa.chunked_array([], ty)) == []


def test_pyarrow_table(pa: Any) -> None:
    table = pa.table({"text": DOCS, "n": list(range(len(DOCS)))})
    assert pj.lcut(table.column("text")) == per_doc("lcut", list(DOCS))
    with pytest.raises(pyroid.InputError):
        pj.lcut(table)


def test_tokenizer_instances_take_batches() -> None:
    t = pj.Tokenizer()
    t.add_word("石墨烯烯", 1000)
    docs = ["石墨烯烯是什么", *DOCS]
    out = t.lcut(docs, threads=2)
    assert out == [t.lcut(d) for d in docs]
    assert "石墨烯烯" in out[0]
    assert "石墨烯烯" not in pj.lcut(docs[0])  # the default tokenizer is unchanged


# ---- threads -------------------------------------------------------------------------------

BIG = [DOCS[i % len(DOCS)] + str(i) for i in range(300)]  # > 64 docs: the pool is used


@pytest.mark.parametrize("mode", list(MODES))
def test_threads_equivalence(mode: str) -> None:
    f = MODES[mode]
    want = per_doc(mode, list(BIG))
    col = _testing.column(BIG, chunks=4)
    for threads in (None, 1, 2, 4):
        assert f(BIG, threads=threads) == want, threads
        assert f(col, threads=threads) == want, threads


@pytest.mark.thread_unsafe(reason="10k documents")
@pytest.mark.timeout(300)
def test_ten_thousand_documents() -> None:
    docs = [f"{DOCS[i % len(DOCS)]}第{i}号文档。" for i in range(10_000)]
    out = pj.lcut(docs)
    assert len(out) == 10_000
    assert out[1234] == pj.lcut(docs[1234])
    assert out[9999] == pj.lcut(docs[9999])
    toks = list(pj.tokenize(docs, mode="search", threads=4))
    assert toks[5000] == list(pj.tokenize(docs[5000], mode="search"))


# About 30,000 words: the words of a batch this size are interned on the pool in several tasks
# (from 16,384 words, at least 8,192 per task).
LARGE = [DOCS[i % len(DOCS)] * 12 + f"第{i}号" for i in range(400)]


@pytest.mark.parametrize("mode", list(MODES))
def test_large_batches_match_per_document(mode: str) -> None:
    f = MODES[mode]
    values: list[str | None] = [None, *LARGE[:150], None, *LARGE[150:], None]
    want = per_doc(mode, values)
    col = _testing.column(values, chunks=3)
    for threads in (None, 1, 2, 8):
        out = f(LARGE, threads=threads)
        assert out == [w for w in want if w is not None], threads
        assert type(out) is list
        assert all(type(r) is list for r in out)
        assert f(tuple(LARGE), threads=threads) == out, threads
        assert f(col, threads=threads) == want, threads


@pytest.mark.thread_unsafe(reason="about 600,000 words")
@pytest.mark.timeout(300)
def test_many_distinct_words() -> None:
    """150,000 distinct words, each in two far-apart documents (so in two of the pool's ranges)."""
    base = [" ".join(f"w{i}x{j}" for j in range(100)) for i in range(1500)]
    want = [pj.lcut(d) for d in base] * 2
    assert want[0][:3] == ["w0x0", " ", "w0x1"]
    for threads in (1, 2, None):
        assert pj.lcut(base + base, threads=threads) == want, threads


GC_STRESS = """
import gc, threading
import pyroid.jieba as pj

docs = [f"小明硕士毕业于中国科学院计算所，后在日本京都大学深造{i}" * 8 for i in range(300)]
want = {
    "lcut": [pj.lcut(d) for d in docs],
    "lcut_for_search": [pj.lcut_for_search(d) for d in docs],
    "tokenize": [list(pj.tokenize(d)) for d in docs],
}
errors = []
stop = threading.Event()

def worker(k):
    try:
        for it in range(ROUNDS):
            threads = [None, 2, 8, 3][(k + it) % 4]
            for name, f in (("lcut", pj.lcut), ("lcut_for_search", pj.lcut_for_search)):
                if f(docs, threads=threads) != want[name]:
                    errors.append(name)
            if list(pj.tokenize(docs, threads=threads)) != want["tokenize"]:
                errors.append("tokenize")
    except BaseException as e:
        errors.append(f"worker {k}: {type(e).__name__}: {e}")

def collector():
    while not stop.wait(0.001):
        gc.collect()

g = threading.Thread(target=collector)
g.start()
ws = [threading.Thread(target=worker, args=(k,)) for k in range(6)]
for w in ws:
    w.start()
for w in ws:
    w.join()
stop.set()
g.join()
assert not errors, errors[:5]
print("ok")
"""


@pytest.mark.thread_unsafe(reason="runs a subprocess with 7 threads of its own")
@pytest.mark.timeout(240)
def test_concurrent_large_batches_with_gc_do_not_deadlock(run_python: RunPython) -> None:
    r = run_python("ROUNDS = 8\n" + GC_STRESS, timeout=200)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip() == "ok"


def test_bad_threads() -> None:
    for bad in (0, -1):
        with pytest.raises(ValueError, match="threads must be None or a positive int"):
            pj.lcut(DOCS, threads=bad)
    for wrong in ("2", 2.0):
        with pytest.raises(pyroid.InputError):
            pj.lcut(DOCS, threads=wrong)  # type: ignore[call-overload]


# ---- errors ----------------------------------------------------------------------------------


@pytest.mark.parametrize("item", [1, None, b"bytes", 2.5, ["nested"]])
@pytest.mark.parametrize("mode", list(MODES))
def test_non_str_item(mode: str, item: Any) -> None:
    with pytest.raises(pyroid.InputError, match="at index 1"):
        MODES[mode](["好的", item])


@pytest.mark.parametrize("x", [5, {"a": 1}, {"a"}, iter(["a"])])
def test_wrong_input_type(x: Any) -> None:
    for mode in ("lcut", "lcut_no_hmm", "cut_all", "search", "search_no_hmm", "cut"):
        with pytest.raises(pyroid.InputError):
            MODES[mode](x)


def zero_total_tokenizer() -> Any:
    """Return a tokenizer whose dictionary totals 0, so cutting a word block takes ``log(0)``."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dict.txt")
        Path(path).write_bytes("北京 5\n大学 -5\n".encode())
        t = pj.Tokenizer(path)
        t.initialize()
    return t


LOG_ZERO = "math domain error|expected a positive input"  # math.log(0): CPython < 3.14, 3.14+


@pytest.mark.parametrize("bad", [[0], [7, 300], [399], [150, 151]])
def test_lowest_failing_document_is_reported(bad: list[int]) -> None:
    t = zero_total_tokenizer()
    with pytest.raises(ValueError, match=LOG_ZERO) as one:
        t.lcut("北京大学")
    assert not hasattr(one.value, "__notes__")
    docs = ["，" * 100] * 400  # no word block: cut without the dictionary
    for i in bad:
        docs[i] = "北京大学"
    calls: dict[str, Callable[..., Any]] = {
        "lcut": t.lcut,
        "lcut_for_search": t.lcut_for_search,
        "tokenize": lambda x, **kw: list(t.tokenize(x, **kw)),
    }
    note = f"while segmenting the document at index {min(bad)}"
    for name, f in calls.items():
        for threads in (None, 1, 2, 8):
            with pytest.raises(ValueError, match=LOG_ZERO) as info:
                f(docs, threads=threads)
            assert str(info.value) == str(one.value), (name, threads)
            assert info.value.__notes__ == [note], (name, threads)
    assert t.lcut(docs[1:7]) == [["，"] * 100] * 6


def test_tokenize_needs_text() -> None:
    for x in (5, b"bytes", None):
        with pytest.raises(ValueError, match=r"^jieba: the input parameter should be unicode\.$"):
            list(pj.tokenize(x))  # type: ignore[call-overload]


def test_arrow_non_string_column(pa: Any) -> None:
    for f in MODES.values():
        with pytest.raises(pyroid.InputError, match="Arrow string column"):
            f(pa.array([1, 2]))
        with pytest.raises(pyroid.InputError):
            f(pa.array(["a", "b"]).dictionary_encode())


def test_lone_surrogates() -> None:
    with pytest.raises(UnicodeEncodeError) as info:
        pj.lcut("北京\ud800")
    assert any("lone surrogate" in n for n in info.value.__notes__)
    with pytest.raises(UnicodeEncodeError) as info:
        pj.lcut(["北京", "北京\udfff"])
    assert any("index 1" in n for n in info.value.__notes__)
    for f in MODES.values():
        with pytest.raises(UnicodeEncodeError):
            f(["好", "\ud83d"])


# ---- bytes -----------------------------------------------------------------------------------


def test_bytes_are_decoded_as_jieba_does() -> None:
    s = "我来到北京清华大学"
    assert pj.lcut(s.encode("utf-8")) == pj.lcut(s)
    assert pj.lcut(s.encode("gbk")) == pj.lcut(s)  # not UTF-8: GBK, errors ignored
    assert pj.lcut(bytearray(s.encode("utf-8"))) == pj.lcut(s)
    assert pj.lcut_for_search(s.encode("gbk")) == pj.lcut_for_search(s)
    assert pj.lcut(b"\xff\xfeabc") == pj.lcut(b"\xff\xfeabc".decode("gbk", "ignore"))
    assert pj.strdecode(b"\xe4\xb8\xad") == "中"
    assert pj.strdecode(5) == 5


# ---- FREQ ------------------------------------------------------------------------------------


def test_freq_before_and_after_loading() -> None:
    t = pj.Tokenizer()
    assert not t.initialized
    assert isinstance(t.FREQ, collections.abc.Mapping)
    assert len(t.FREQ) == 0
    assert t.total == 0
    assert "北京" not in t.FREQ
    assert t.FREQ.get("北京") is None
    assert t.FREQ.get("北京", 7) == 7
    with pytest.raises(KeyError):
        t.FREQ["北京"]
    assert list(t.FREQ) == []

    t.initialize()
    assert t.initialized
    assert len(t.FREQ) == 498_113
    assert t.total == 60_101_967
    assert "北京" in t.FREQ
    assert t.FREQ["北京"] == t.FREQ.get("北京") == 34_488
    assert t.FREQ["清华大学"] == 922
    assert t.FREQ["清华大"] == 0  # a prefix of a word
    assert t.FREQ["AT&T"] == 3
    assert t.FREQ["AT&"] == 0
    assert "清华大学x" not in t.FREQ
    assert t.FREQ.get("清华大学x", -1) == -1
    for key in (5, None, b"\xe5\x8c\x97"):
        assert key not in t.FREQ
        assert t.FREQ.get(key) is None
        with pytest.raises(KeyError):
            t.FREQ[key]
    with pytest.raises(TypeError):
        t.FREQ["北京"] = 1


def test_freq_iteration() -> None:
    t = pj.Tokenizer()
    t.initialize()
    t.add_word("石墨烯烯烯", 7)
    keys = list(t.FREQ)
    # 石墨 is in the dictionary; 石墨烯, 石墨烯烯 and 石墨烯烯烯 are new (as in jieba).
    assert len(keys) == len(set(keys)) == len(t.FREQ) == 498_113 + 3
    # Documented order: the dictionary's keys in UTF-8 byte order, then the added ones.
    assert keys[-3:] == ["石墨烯", "石墨烯烯", "石墨烯烯烯"]
    base = keys[:-3]
    assert base == sorted(base, key=lambda k: k.encode())
    assert dict(t.FREQ.items())["石墨烯烯烯"] == 7


def test_module_freq_and_get_freq() -> None:
    pj.dt.check_initialized()
    assert len(pj.dt.FREQ) == 498_113
    assert pj.get_FREQ("北京") == 34_488
    assert pj.get_FREQ("清华大学x") is None
    assert pj.get_FREQ("清华大学x", 0) == 0
    assert pj.user_word_tag_tab is pj.dt.user_word_tag_tab


def test_freq_matches_jieba_len() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jieba: Any = pytest.importorskip("jieba")
    assert len(jieba.Tokenizer().FREQ) == len(pj.Tokenizer().FREQ) == 0
    t = jieba.Tokenizer()
    t.check_initialized()
    assert len(t.FREQ) == len(pj.dt.FREQ) == 498_113


def test_add_word_tags_and_total() -> None:
    t = pj.Tokenizer()
    t.add_word("石墨烯烯", 5, "nz")
    assert t.initialized
    assert t.user_word_tag_tab == {"石墨烯烯": "nz"}
    assert t.total == 60_101_967 + 5
    assert t.FREQ["石墨烯烯"] == 5
    t.add_word("石墨烯烯", "7")
    assert t.FREQ["石墨烯烯"] == 7
    assert t.total == 60_101_967 + 12
    with pytest.raises(ValueError, match="invalid literal for int"):
        t.add_word("x", "seven")


# ---- enable_parallel --------------------------------------------------------------------------


@pytest.mark.thread_unsafe(reason="enable_parallel replaces module globals")
@pytest.mark.parametrize("processnum", [None, 1, 4])
def test_enable_parallel(processnum: int | None) -> None:
    """Parallel mode cuts each ``splitlines(True)`` piece, as jieba's ``_pcut`` does: the same
    words as a whole-text cut, except that full mode then yields its empty strings per line."""
    text = "\n".join(DOCS) + "\r\n最后一行 分隔\x85再一行\n\n"
    lines = text.splitlines(True)
    want = {
        "cut": pj.lcut(text),
        "cut_no_hmm": pj.lcut(text, HMM=False),
        "cut_all": [w for ln in lines for w in pj.lcut(ln, cut_all=True)],
        "search": pj.lcut_for_search(text),
        "search_no_hmm": pj.lcut_for_search(text, HMM=False),
    }
    assert want["cut"] == [w for ln in lines for w in pj.lcut(ln)]
    assert want["cut_all"] != pj.lcut(text, cut_all=True)  # "\n\n": per line, more ''
    pj.enable_parallel(processnum)
    try:
        assert pj.cut is pj._pcut
        assert pj.cut_for_search is pj._pcut_for_search
        got = {
            "cut": list(pj.cut(text)),
            "cut_no_hmm": list(pj.cut(text, HMM=False)),
            "cut_all": list(pj.cut(text, cut_all=True)),
            "search": list(pj.cut_for_search(text)),
            "search_no_hmm": list(pj.cut_for_search(text, HMM=False)),
        }
        assert got == want
        assert list(pj.cut(text.encode("utf-8"))) == want["cut"]
        assert pj.lcut(text) == want["cut"]  # lcut is not parallel in jieba either
    finally:
        pj.disable_parallel()
    assert pj.cut == pj.dt.cut
    assert pj.cut_for_search == pj.dt.cut_for_search
    assert list(pj.cut(text)) == want["cut"]


def test_enable_parallel_matches_jieba_split() -> None:
    """jieba's parallel mode cuts ``splitlines(True)`` pieces; so does pyroid's."""
    text = "第一行\r\n第二行\r第三行\n\n第五行\u2029第六行\x0c"
    parts = text.splitlines(True)
    want = [w for p in parts for w in pj.lcut(p)]
    assert list(pj._pcut(text)) == want
    assert list(pj._pcut_for_search(text)) == [w for p in parts for w in pj.lcut_for_search(p)]


# ---- the delegated parts (child interpreters) ------------------------------------------------

# The same changes, made through jieba (the reference) and through pyroid.jieba: add_word with
# a tag, load_userdict, delegated calls, more changes, more delegated calls. jieba.posseg moves
# jieba.dt.user_word_tag_tab into its own table on each call, so the tag table is only
# compared before the first one.
DELEGATE_SCRIPT = r"""
import json, logging, os, sys, tempfile, warnings
warnings.simplefilter("error")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # jieba's SyntaxWarnings on first compile
    import jieba, jieba.posseg, jieba.analyse
jieba.setLogLevel(logging.WARNING)
CANDIDATE = os.environ["PYROID_TEST_WHO"] == "pyroid"
if CANDIDATE:
    import pyroid
    import pyroid.jieba as J
    import pyroid.jieba.posseg as POS
    import pyroid.jieba.analyse as AN
else:
    J, POS, AN = jieba, jieba.posseg, jieba.analyse
text = "李小福是创新办主任也是云计算方面的专家。石墨烯烯是一种新材料，北京大学的研究生在研究它。"
d = tempfile.mkdtemp()
userdict = os.path.join(d, "userdict.txt")
with open(userdict, "wb") as f:
    f.write("云计算 5\n李小福 2 nr\n创新办 3 i\n".encode())
out = {}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    J.add_word("石墨烯烯", 50, "nz")
    with open(userdict, "rb") as f:
        J.load_userdict(f)
    out["tag_tab"] = sorted(J.user_word_tag_tab.items())
    out["pos1"] = [(p.word, p.flag) for p in POS.lcut(text)]
    out["pos1_gen"] = [(p.word, p.flag) for p in POS.cut(text)]
    out["pos1_no_hmm"] = [(p.word, p.flag) for p in POS.lcut(text, HMM=False)]
    J.del_word("研究生")
    J.add_word("新材料", tag="n")
    J.suggest_freq(("北京", "大学"), True)
    out["pos2"] = [(p.word, p.flag) for p in POS.lcut(text)]
    out["tags"] = AN.extract_tags(text, topK=8)
    out["tags_weights"] = AN.extract_tags(text, topK=8, withWeight=True)
    out["textrank"] = AN.textrank(text, topK=8)
    out["textrank_weights"] = AN.textrank(text, topK=8, withWeight=True)
    out["cut"] = J.lcut(text)
    t = J.Tokenizer()
    t.add_word("石墨烯烯", 50, "nz")
    t.del_word("研究生")
    out["pos_tokenizer"] = [(p.word, p.flag) for p in POS.POSTokenizer(t).lcut(text)]
    f = J.get_dict_file()
    out["dict_head"] = f.readline().decode()
    f.close()
out["fallback"] = sorted(
    str(w.message) for w in caught if CANDIDATE and w.category is pyroid.FallbackWarning
)
out["other_warnings"] = sorted(
    w.category.__name__ for w in caught if not (CANDIDATE and w.category is pyroid.FallbackWarning)
)
print(json.dumps(out, ensure_ascii=False))
"""


@pytest.mark.thread_unsafe(reason="subprocesses")
def test_delegated_submodules_match_jieba(run_python: RunPython) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pytest.importorskip("jieba")
    results = {}
    for who in ("jieba", "pyroid"):
        r = run_python(DELEGATE_SCRIPT, env={"PYROID_TEST_WHO": who}, timeout=300.0)
        assert r.returncode == 0, r.stderr
        results[who] = json.loads(r.stdout)
    got, want = results["pyroid"], results["jieba"]
    fallback = got.pop("fallback")
    assert want.pop("fallback") == []
    assert got == want
    assert got["other_warnings"] == []
    assert ["石墨烯烯", "nz"] in got["pos1"]
    assert ["李小福", "nr"] in got["pos1"]
    assert ["新材料", "n"] in got["pos2"]
    # One FallbackWarning per delegated feature, however many calls.
    assert len(fallback) == 3, fallback
    assert sum("jieba.posseg" in m for m in fallback) == 1, fallback
    assert sum("jieba.analyse" in m for m in fallback) == 1, fallback
    assert sum("get_dict_file" in m for m in fallback) == 1, fallback


NO_JIEBA_SCRIPT = r"""
import sys
sys.modules["jieba"] = None  # import jieba now raises ImportError
import pyroid, pyroid.jieba as pj, pyroid.jieba.posseg as pseg, pyroid.jieba.analyse as an
assert pj.lcut("我来到北京清华大学") == ["我", "来到", "北京", "清华大学"]
pj.add_word("石墨烯烯", 10)
assert "石墨烯烯" in pj.lcut("石墨烯烯")
calls = {
    "posseg.lcut": lambda: pseg.lcut("我来到北京"),
    "posseg.cut": lambda: list(pseg.cut("我来到北京")),
    "analyse.extract_tags": lambda: an.extract_tags("我来到北京"),
    "analyse.textrank": lambda: an.textrank("我来到北京"),
    "get_dict_file": pj.get_dict_file,
    "enable_paddle": pj.enable_paddle,
}
for name, call in calls.items():
    try:
        call()
    except pyroid.PyroidError as e:
        assert "pip install jieba==0.42.1" in str(e), (name, str(e))
        assert isinstance(e.__cause__, ImportError), name
    else:
        raise AssertionError(name + " did not raise")
print("ok")
"""

NOT_IMPORTED_SCRIPT = r"""
import sys
import pyroid.jieba as pj
pj.lcut("我来到北京清华大学"); pj.lcut_for_search("北京"); list(pj.tokenize("北京"))
pj.add_word("石墨烯烯"); pj.del_word("北京"); pj.suggest_freq(("北", "京"), True)
t = pj.Tokenizer(); t.lcut(["北京", "大学"], threads=2)
print("jieba" in sys.modules, sorted(m for m in sys.modules if m.startswith("jieba")))
"""


@pytest.mark.thread_unsafe(reason="subprocesses")
def test_pyroid_error_without_jieba(run_python: RunPython) -> None:
    r = run_python(NO_JIEBA_SCRIPT)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"


@pytest.mark.thread_unsafe(reason="subprocesses")
def test_segmentation_does_not_import_jieba(run_python: RunPython) -> None:
    r = run_python(NOT_IMPORTED_SCRIPT)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False []"


@pytest.mark.thread_unsafe(reason="subprocesses")
def test_gil_stays_disabled(run_python: RunPython) -> None:
    if not FREE_THREADED:
        pytest.skip("a GIL build")
    code = """
    import sys, warnings
    import pyroid.jieba as pj
    pj.lcut(["我来到北京清华大学"] * 200, threads=4)
    pj.add_word("石墨烯烯")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import jieba
        jieba.lcut("我来到北京清华大学")
    except ImportError:
        pass
    print(sys._is_gil_enabled())
    """
    r = run_python(code)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"


# ---- module surface ----------------------------------------------------------------------


def test_jieba_names() -> None:
    for name in pj.__all__:
        assert hasattr(pj, name), name
    assert pj.JIEBA_VERSION == pj.__version__ == "0.42.1"
    assert pj.DEFAULT_DICT is None
    assert pj.DEFAULT_DICT_NAME == "dict.txt"
    assert repr(pj.dt) == "<Tokenizer dictionary=None>"


def test_jieba_patterns() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jieba: Any = pytest.importorskip("jieba")
    for name in ("re_userdict", "re_eng", "re_han_default", "re_skip_default"):
        assert getattr(pj, name).pattern == getattr(jieba, name).pattern, name
        assert getattr(pj, name).flags == getattr(jieba, name).flags, name


def test_delegated_submodules_answer_dunder_lookups_without_jieba() -> None:
    """hasattr, help() and star imports on posseg/analyse never import (or need) jieba."""
    import pydoc

    import pyroid.jieba.analyse as an
    import pyroid.jieba.posseg as pseg

    for mod, names in ((pseg, {"cut", "lcut", "POSTokenizer"}), (an, {"extract_tags", "textrank"})):
        assert not hasattr(mod, "__wrapped__")  # no PyroidError, no FallbackWarning
        assert names <= set(mod.__all__)
        assert pydoc.render_doc(mod)
        ns: dict[str, object] = {}
        exec(f"from {mod.__name__} import *", ns)
        assert names <= ns.keys()
