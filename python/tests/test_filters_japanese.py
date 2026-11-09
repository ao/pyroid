"""Japanese in pyroid.filters and pyroid.dedup: datatrove's ``SpaCyTokenizer("ja")``.

datatrove tokenizes the codes ``ja``, ``ja_Jpan``, ``jpn`` and ``jpn_Jpan`` with spaCy's Japanese
pipeline on SudachiPy (split mode A, SudachiDict-core), in chunks of at most 40,000 UTF-8 bytes.
pyroid runs its port of that when it finds a dictionary (the installed ``sudachidict_core``
package, or the file ``$PYROID_SUDACHI_DICT`` names); without one the codes need
``python_tokenizer=True`` like the other non-native codes.

- The language table, native or opt-in as the dictionary allows (no datatrove needed).
- Live datatrove (skips without datatrove, spaCy, SudachiPy and SudachiDict-core): Hypothesis
  Japanese documents, short and over the 40,000-byte chunk size (some with a lone space ending
  a chunk, which shifts datatrove's later sentence spans by one character), through every
  filter and chains, word lists, sentence splits, MinHash and the sentence-dedup stages.
- Container kinds, Arrow, ``threads=``, a 2,000-document batch, input errors, lone surrogates
  and SudachiPy's input limit.
- The dictionary in child interpreters: ``$PYROID_SUDACHI_DICT`` naming a missing file or not a
  dictionary, no dictionary at all, a dictionary only SudachiPy finds, the first use from
  several threads.

``PYROID_FILTERS_EXAMPLES=n`` sets the Hypothesis examples per test.
"""

# Look-alike characters and other scripts on purpose.
# ruff: noqa: RUF001

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import random
import string
import struct
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument

import pyroid
import pyroid.dedup as dd
import pyroid.filters as pf
from pyroid._pyroid import _testing

pytestmark = [
    # datatrove's and spaCy's tokenizers are not thread-safe; the warning is once per process.
    pytest.mark.thread_unsafe(reason="datatrove, spaCy and the once-per-process warning"),
    # Which path ran, not what the result is (MODULES.md); parity suites ignore it.
    pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::FutureWarning"),
]

_HERE = Path(__file__).parent


def _sibling(name: str) -> Any:
    """A sibling test module, imported by path (the suite runs without python/tests on sys.path)."""
    spec = importlib.util.spec_from_file_location(f"_pyroid_ja_{name}", _HERE / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: The filters' harness (Datatrove, candidate, configs, CHAINS, fineweb2_steps), MinHash's
#: (Datatrove, check) and the sentence dedup's (run_stages, dt_config, py_config, norm_configs).
FILTERS = _sibling("test_filters_multilingual")
DEDUP = _sibling("test_dedup_multilingual")
SENT = _sibling("test_sentence_dedup_more")

CODES = ["ja", "ja_Jpan", "jpn", "jpn_Jpan"]
JA = "jpn_Jpan"
CHUNK = 40_000  # SpaCyTokenizer._do_tokenize: chunk_text_on_bytes(text, 40000) for "ja"
STOPS: list[str] = FILTERS.FINEWEB2[JA]["stopwords"]


def sudachi_dictionary() -> bool:
    """Whether pyroid finds a SudachiDict as it looks for one (Japanese is then native)."""
    if "PYROID_SUDACHI_DICT" in os.environ:
        return True
    spec = importlib.util.find_spec("sudachidict_core")
    return spec is not None and spec.origin is not None


DICTIONARY = sudachi_dictionary()
needs_dictionary = pytest.mark.skipif(
    not DICTIONARY, reason="no SudachiDict (pip install sudachidict-core): Japanese is not native"
)


def examples(n: int) -> settings:
    env = os.environ.get("PYROID_FILTERS_EXAMPLES")
    return settings(
        max_examples=int(env) if env else n, deadline=None, suppress_health_check=list(HealthCheck)
    )


def _notes(exc: BaseException) -> str:
    return "\n".join(getattr(exc, "__notes__", []))


# ---- Japanese documents -------------------------------------------------------------

HIRAGANA = [
    "は", "の", "に", "を", "が", "と", "で", "です", "ます", "した", "ない", "こんにちは",
    "ありがとう", "すごーい", "すご〜い", "ゝゞ", "ゟ", "っ", "ぁぃぅ", "が", "ぱ",
    "でしょう", "ください", "とても",
]  # fmt: skip
KATAKANA = [
    "カタカナ", "コーヒー", "ラーメン", "ヴァイオリン", "テスト", "ッ", "ヵヶ", "ヷヸ", "・",
    "ア・イ", "ヽヾ", "ｶﾀｶﾅ", "ｺｰﾋｰ", "ﾃﾞｰﾀ", "ｶﾞｷﾞｸﾞ", "ﾊﾟﾋﾟﾌﾟ", "ｳﾞｧｲｵﾘﾝ", "ｶ゙", "ｶ゛",
    "ｶﾟﾞ", "ｰ", "ﾞ", "ﾟ", "ｱｲｳｴｵ", "ァア", "ンンン",
]  # fmt: skip
KANJI = [
    "日本語", "東京", "首都", "学生", "私", "漢字", "公園", "散歩", "天気", "株式会社",
    "東京都", "大阪", "々", "人々", "〆", "ヶ月", "葛︀", "𠀀", "鶴", "年", "月", "日",
]  # fmt: skip
NUMERALS = [
    "〇", "一", "十", "十一", "二十", "百", "千", "万", "一万", "一億", "二〇二四年",
    "三十五万六千七百八十九", "千百十", "1万2000", "1億2千万", "1.5万", "第1,234回", "4十",
]  # fmt: skip
DIGITS = [
    "0", "007", "2024", "1,000", "1,000,000.5", "3.14", "12,34", ",123", "1,", "1.2.3",
    "1..2", "１２３", "２，０００", "１．５", "１，０００，０００", "10万円", "3.5%",
    "192.168.0.1", "9" * 30, "½", "²", "1/2", "1:30",
]  # fmt: skip
PROLONGED = [
    "ー", "ーー", "ーーー", "〜", "〜〜", "～", "-", "--", "---", "⁓", "〰", "ラーーメン",
    "すごーーい", "長ーーーい", "ｰｰ", "－－", "ー-〜", "カーー", "ーア",
]  # fmt: skip
YOMIGANA = [
    "漢字(かんじ)", "東京（とうきょう）", "日本(にほん)", "読み（よみ）", "(かな)", "（カナ）",
    "字(じ", "字（", "（）", "()", "東京(Tokyo)", "日本（２０２４）",
]  # fmt: skip
PUNCT = [
    "。", "、", "！", "？", "「", "」", "『", "』", "（", "）", "…", "‥", "!", "?", ".", ",",
    ":", "；", "※", "♪", "★", "【", "】", "｡", "､", "･", "｢", "｣", "“", "”", "...", "!!",
]  # fmt: skip
EMOJI = ["😀", "👍🏽", "🇯🇵", "👨‍👩‍👧", "❤️", "1️⃣", "🀄", "🫠"]
OTHER = [
    "Python", "iPhone 15 Pro", "COVID-19", "e.g.", "Hello. World.", "C++",
    "https://example.com/a?b=日本", "info@example.co.jp", "ΑΒΓ", "αβγ", "ΣΑΣ", "Москва",
    "Привет", "İstanbul", "ẞ", "IS_ALPHA", "LENGTH", "PUNCT", "ＡＢＣ", "ａｂｃ",
    "Ｔｏｋｙｏ２０２０",
    "javascript", "lorem ipsum", "{", "}", "[1]", "#tag", "@user",
]  # fmt: skip
#: Characters SudachiPy's normalization makes longer (NFKC and its rewrite rules).
EXPANDING = [
    "㍿", "…", "①", "⑳", "㉑", "Ⅻ", "ⅰⅱ", "㌀", "㌔", "㍻", "㋿", "℡", "№", "ﷺ", "ﬃ", "⑽",
    "⒇", "㈱", "℃", "㏍",
]  # fmt: skip
WHITESPACE = [" "] * 6 + [
    "  ", "   ", "　", "　　", "\t", "\xa0", " ", "​", "\x0c", "\x0b",
    " ", "﻿", " ", " ", " 　 ",
]  # fmt: skip
LINE_SEPS = ["\n"] * 8 + ["\n\n"] * 3 + ["\r\n", "\r", "\n \n", "\x85", " ", "\n　"]
ENDINGS = ["。", "。", "。", "！", "？", "", ".", "…", "」", "。」", "！？", "?!", "。 ", "）"]
PIECES = HIRAGANA + KATAKANA + KANJI + NUMERALS + DIGITS + PROLONGED + YOMIGANA + PUNCT + OTHER
_CHARS = "".join(sorted({c for p in PIECES + EXPANDING + EMOJI for c in p}))
_KANA_KANJI = st.one_of(
    st.characters(min_codepoint=0x3040, max_codepoint=0x30FF),
    st.characters(min_codepoint=0xFF61, max_codepoint=0xFF9F),
    st.characters(min_codepoint=0x4E00, max_codepoint=0x4FFF),
)


def pieces() -> st.SearchStrategy[str]:
    return st.one_of(
        st.sampled_from(HIRAGANA),
        st.sampled_from(HIRAGANA),
        st.sampled_from(KANJI),
        st.sampled_from(KANJI),
        st.sampled_from(KATAKANA),
        st.sampled_from(NUMERALS + DIGITS),
        st.sampled_from(PROLONGED + YOMIGANA),
        st.sampled_from(PUNCT),
        st.sampled_from(EMOJI + EXPANDING),
        st.sampled_from(OTHER),
        st.text(st.sampled_from(_CHARS), min_size=1, max_size=6),
        st.text(_KANA_KANJI, min_size=1, max_size=6),
        st.text(st.characters(exclude_categories=["Cs"]), min_size=1, max_size=3),
    )


@st.composite
def lines(draw: st.DrawFn) -> str:
    """Japanese pieces, mostly unspaced, sometimes with any kind of whitespace between them."""
    ps = draw(st.lists(pieces(), max_size=24))
    seps = draw(
        st.lists(st.sampled_from([""] * 14 + WHITESPACE), min_size=len(ps), max_size=len(ps))
    )
    line = "".join(s + p for s, p in zip(seps, ps, strict=True)) + draw(st.sampled_from(ENDINGS))
    if draw(st.integers(0, 7)) == 0:
        line = draw(st.sampled_from(WHITESPACE)) + line
    return line


@st.composite
def documents(draw: st.DrawFn) -> str:
    """0 to ~16 lines, sometimes repeated (for the repetition checks)."""
    shape = draw(st.integers(0, 9))
    if shape == 0:
        return draw(st.sampled_from(["", " ", "\n", "　", "。", "ー", "㍿", " \n "]))
    if shape == 1:  # one phrase over and over
        phrase = "".join(draw(st.lists(pieces(), min_size=1, max_size=6)))
        sep = draw(st.sampled_from(["", " ", "\n", "。", "。\n", "　", "\n\n"]))
        return sep.join([phrase] * draw(st.integers(1, 30)))
    ls = draw(st.lists(lines(), min_size=1, max_size=16))
    picks = list(range(len(ls)))
    for _ in range(draw(st.sampled_from([0, 0, 1, 3, 8]))):
        picks.insert(draw(st.integers(0, len(picks))), draw(st.integers(0, len(ls) - 1)))
    seps = draw(st.lists(st.sampled_from(LINE_SEPS), min_size=len(picks), max_size=len(picks)))
    doc = "".join(ls[i] + s for i, s in zip(picks, seps, strict=True))
    return doc.rstrip("\n") if draw(st.booleans()) else doc


#: 1-byte characters that are not whitespace: swapping one for another, or for a space or
#: "\n", keeps every chunk boundary of chunk_text_on_bytes where it is.
FILL = string.ascii_lowercase + string.digits


def chunk_ends(text: str, chunk: Callable[[str, int], list[str]]) -> list[int]:
    """Where datatrove's chunks of ``text`` end (character offsets), the last one left out."""
    ends, at = [], 0
    for c in chunk(text, CHUNK)[:-1]:
        at += len(c)
        ends.append(at)
    return ends


def at_chunk_ends(text: str, how: str, chunk: Callable[[str, int], list[str]]) -> str:
    """``text`` with ASCII around each chunk boundary, then ``how`` whitespace at it.

    ``"space"`` makes a lone space the last character of the chunk (datatrove then starts the
    next chunk's sentence spans one character early), ``"spaces"`` two spaces, ``"newline"``
    a line break, ``"space-after"`` a space as the first character of the next chunk.
    """
    chars = list(text)

    def ascii_at(e: int) -> bool:
        return all(c in FILL for c in chars[max(e - 12, 0) : e + 4])

    for _ in range(12):  # each pass puts one more boundary in ASCII
        bad = [e for e in chunk_ends("".join(chars), chunk) if not ascii_at(e)]
        if not bad:
            break
        e = bad[0]
        lo, hi = max(e - 60, 0), min(e + 400, len(chars))
        chars[lo:hi] = [FILL[i % len(FILL)] for i in range(lo, hi)]
    for e in chunk_ends("".join(chars), chunk):
        if not ascii_at(e):
            continue
        if how == "space":
            chars[e - 1] = " "
        elif how == "spaces":
            chars[e - 2] = chars[e - 1] = " "
        elif how == "newline":
            chars[e - 1] = "\n"
        elif how == "space-after":
            chars[e] = " "
    return "".join(chars)


@st.composite
def long_documents(draw: st.DrawFn, chunk: Callable[[str, int], list[str]]) -> str:
    """Over 40,000 bytes: drawn lines repeated (as lines, or one long line), numbered now and
    then, often with a lone space ending each chunk."""
    base = draw(st.lists(lines(), min_size=1, max_size=6))
    one_line = draw(st.booleans())
    sep = st.sampled_from(["", *WHITESPACE]) if one_line else st.sampled_from(LINE_SEPS)
    seps = draw(st.lists(sep, min_size=len(base), max_size=len(base)))
    unit = "".join(b + s for b, s in zip(base, seps, strict=True))
    if not unit.strip():
        unit += "日本語の文です。"
    size = draw(st.integers(CHUNK + 1, 3 * CHUNK + 500))
    n = size // len(unit.encode()) + 1
    text = "".join(unit if i % 7 else f"{unit}{i}番目。" for i in range(n))
    how = draw(st.sampled_from(["", "space", "space", "space", "spaces", "newline", "space-after"]))
    if how:
        text = at_chunk_ends(text, how, chunk)
    pieces_ = chunk(text, CHUNK)
    lone = sum(c.endswith(" ") and not c.endswith("  ") for c in pieces_[:-1])
    _event(f"{len(pieces_)} chunks, {lone} ending in a lone space")
    return text


# ---- the language table and the dictionary ----------------------------------------------


def test_japanese_codes() -> None:
    langs = pf.languages()
    assert [
        c for c, v in langs.items() if (v.tokenizer, v.arg) == ("SpaCyTokenizer", "ja")
    ] == CODES
    for code in CODES:
        assert langs[code] == pf.Language("SpaCyTokenizer", "ja", DICTIONARY), code
        assert dd.languages()[code].native is DICTIONARY
    assert sum(v.native for v in langs.values()) == (4013 if DICTIONARY else 4009)


DOCS = [
    "東京は日本の首都です。人口は約1,400万人で、２０２４年も増えました！\n今日はいい天気ですね。",
    "",
    "短い",
    "ラーメンを食べた。" * 20,
    "\n".join(["この行は十分に長くて、数えるのに十分な文字があります。"] * 6),
    "猫が寝ている。雨が降った。とても寒い。終わり。\n" * 3 + "別のもの。",
    "漢字(かんじ)とｶﾀｶﾅとカタカナ、それに㍿や①やⅫも。　全角スペースも ある。 😀",
    "Python と Rust でプログラムを書きます。 e.g. this… IS_ALPHA",
]


def chain(lang: Any = JA) -> list[Any]:
    """FineWeb-2's Japanese filters, and C4 (which rewrites the text) in front."""
    return [
        pf.C4Quality(language=lang, min_num_sentences=1, filter_no_terminal_punct=False),
        *(step._replace(language=lang) for step in FILTERS.fineweb2_steps(JA)),
    ]


ALL_FUNCS: list[Callable[..., Any]] = FILTERS.ALL_FUNCS


def test_native_with_a_dictionary_else_opt_in() -> None:
    if DICTIONARY:
        want = pf.apply(DOCS, chain())
        for code in CODES:
            assert pf.apply(DOCS, chain(code)) == want, code
            for f in ALL_FUNCS:
                assert f(DOCS, language=code) == f(DOCS, language=JA), (f, code)
        return
    for f in ALL_FUNCS:
        with pytest.raises(NotImplementedError) as e:
            f(["x"], language=JA)
        for name in [
            'SpaCyTokenizer("ja")',
            "SudachiPy",
            "sudachidict-core",
            "python_tokenizer=True",
        ]:
            assert name in str(e.value), (f, name)
    with pytest.raises(NotImplementedError, match="python_tokenizer=True"):
        pf.apply([], chain())
    with pytest.raises(NotImplementedError, match="MinHash still run in Rust"):
        dd.minhash_signatures(["x"], language=JA)
    with pytest.raises(NotImplementedError, match="language 'jpn_Jpan'"):
        dd.SentenceDedup(language=JA)
    # Lines need no tokenizer.
    d = dd.SentenceDedup(language=JA, split_sentences=False, **SENT.NO_MIN)
    assert d.fit(["a\nb\nc", "a\nb\nc"]) == (["a\nb\nc", ""], [True, False])


# ---- live datatrove ------------------------------------------------------------------

#: ("ok", keep, reason, text) or ("raise", exception type name).
Outcome = tuple[Any, ...]


class Reference:
    """datatrove's Japanese tokenizer and filters (test_filters_multilingual's harness)."""

    def __init__(self) -> None:
        self.dt = FILTERS.Datatrove()
        self.tok: Any = self.dt.wt.load_word_tokenizer(JA)
        self.chunk: Callable[[str, int], list[str]] = self.dt.wt.chunk_text_on_bytes

    def run(self, steps: Sequence[Any], text: str) -> Outcome:
        out: Outcome = self.dt.run(steps, text)
        # SudachiPy's input limit: pyroid raises ValueError (sudachipy's SudachiError is not
        # pyroid's to raise).
        return ("raise", "ValueError") if out == ("raise", "SudachiError") else out


@pytest.fixture(scope="module")
def ref() -> Reference:
    if not DICTIONARY:
        pytest.skip("no SudachiDict: Japanese is not native (the opt-in path is tested above)")
    pytest.importorskip("sudachipy")
    pytest.importorskip("sudachidict_core")  # datatrove's tokenizer needs the package itself
    return Reference()


def _short(o: Outcome) -> Outcome:
    if o[0] == "ok" and isinstance(o[3], str) and len(o[3]) > 120:
        return (*o[:3], f"<{len(o[3])} chars> {o[3][:60]!r}...{o[3][-40:]!r}")
    return o


def _first_difference(a: str, b: str) -> str:
    i = next((k for k, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
    lo = max(i - 30, 0)
    return f"at {i} of {len(a)}/{len(b)}: {a[lo : i + 30]!r} vs {b[lo : i + 30]!r}"


def _event(value: str) -> None:
    with contextlib.suppress(InvalidArgument):  # outside a Hypothesis test
        event(value)


def check(ref: Reference, steps: Sequence[Any], docs: list[str]) -> None:
    """pyroid's results for ``docs`` (batched, else per document) are datatrove's."""
    want = [ref.run(steps, d) for d in docs]
    for w in want:
        _event(f"{type(steps[-1]).__name__}: {w[2] if w[0] == 'ok' else w[1]}")
    if all(w[0] == "ok" for w in want):
        res = pf.apply(docs, steps)
        got = [("ok", *r) for r in zip(*res, strict=True)]
    else:
        got = [FILTERS.candidate(steps, d, False) for d in docs]
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        if g != w:
            where = ""
            if g[0] == w[0] == "ok" and g[3] != w[3]:
                where = f"\ntexts differ {_first_difference(g[3], w[3])}"
            pytest.fail(
                f"{steps}\ndocument {i} ({len(docs[i].encode())} bytes): {docs[i][:400]!r}\n"
                f"pyroid {_short(g)} != datatrove {_short(w)}{where}"
            )


def configs(kind: str) -> list[Any]:
    """test_filters_multilingual's spread, FineWeb-2's Japanese settings, and a C4 that keeps
    every sentence (its output shows where datatrove's sentence spans end)."""
    out: list[Any] = FILTERS.configs(kind, JA, STOPS[:6])
    fw2 = dict(zip(["gr", "fw", "gq"], FILTERS.fineweb2_steps(JA), strict=True))
    if kind in fw2:
        out.append(fw2[kind])
    if kind == "c4":
        keep_all: dict[str, Any] = {
            "filter_no_terminal_punct": False,
            "min_num_sentences": -1,
            "min_words_per_line": -1,
            "max_word_length": -1,
            "filter_lorem_ipsum": False,
            "filter_javascript": False,
            "filter_curly_bracket": False,
            "filter_policy": False,
        }
        out.append(pf.C4Quality(language=JA, split_paragraph=False, **keep_all))
        out.append(pf.C4Quality(language=JA, split_paragraph=False, min_words_per_line=2))
    return out


KINDS = ["gq", "gr", "fw", "c4"]


@pytest.mark.parametrize("kind", KINDS)
@examples(80)
@given(data=st.data())
def test_filters_match_datatrove(ref: Reference, kind: str, data: st.DataObject) -> None:
    cfg = data.draw(st.sampled_from(configs(kind)), label="config")
    docs = data.draw(st.lists(documents(), min_size=1, max_size=4), label="docs")
    check(ref, [cfg], docs)


@examples(60)
@given(data=st.data())
def test_chains_match_datatrove(ref: Reference, data: st.DataObject) -> None:
    steps = data.draw(
        st.sampled_from(
            [chain(), FILTERS.fineweb2_steps(JA)] + [c(JA, STOPS) for c in FILTERS.CHAINS]
        )
    )
    docs = data.draw(st.lists(documents(), min_size=1, max_size=4), label="docs")
    check(ref, steps, docs)


@pytest.mark.parametrize("kind", KINDS)
@examples(10)
@given(data=st.data())
def test_long_documents_match_datatrove(ref: Reference, kind: str, data: st.DataObject) -> None:
    cfg = data.draw(st.sampled_from(configs(kind)), label="config")
    doc = data.draw(long_documents(ref.chunk), label="doc")
    check(ref, [cfg], [doc])


def test_a_lone_space_ending_a_chunk_shifts_the_spans_as_datatrove(ref: Reference) -> None:
    unit = "東京は日本の首都です。今日は晴れ。人口は約1,400万人です！\n"
    text = at_chunk_ends(unit * 2000, "space", ref.chunk)
    pieces_ = ref.chunk(text, CHUNK)
    assert len(pieces_) >= 3
    assert all(c.endswith(" ") and not c.endswith("  ") for c in pieces_[:-1])
    spans = ref.tok.span_tokenize(text)
    assert spans[-1][1] < len(text) - 1  # datatrove's later spans start early
    for cfg in configs("c4"):
        check(ref, [cfg], [text, text.replace(" ", "x")])
    same_words(ref, text)


OFF = dd.TextNormConfig(
    lowercase=False,
    norm_whitespace=False,
    remove_punctuation=False,
    norm_unicode_diacritics=False,
    norm_numbers=False,
)


def same_words(ref: Reference, text: str) -> None:
    """pyroid's words for ``text.strip()`` are datatrove's, in order.

    Without normalization ``simplify_text`` only strips, and a MinHash with ``n_grams`` the
    number of words has one shingle: the hash of all the words joined.
    """
    words = ref.tok.word_tokenize(text.strip())
    cfg = dd.MinhashConfig(n_grams=max(len(words), 1), norm_config=OFF, num_buckets=1)
    got = dd.minhash_signatures(text, cfg, language=JA)
    assert got == dd.minhash_signatures(text, cfg, language=ref.tok), (len(words), text[:200])


@examples(150)
@given(doc=documents())
def test_words_match_datatrove(ref: Reference, doc: str) -> None:
    same_words(ref, doc)


@examples(12)
@given(data=st.data())
def test_words_of_long_documents_match_datatrove(ref: Reference, data: st.DataObject) -> None:
    same_words(ref, data.draw(long_documents(ref.chunk), label="doc"))


def sentence_hashes(texts: list[str], n: int, norm: Any) -> None:
    """Stage 1's (hash, sentence) pairs, from ``sent_tokenize``, are datatrove's."""
    kw: dict[str, Any] = {"n_sentences": n, "split_sentences": True}
    step = SENT.dt_sd.SentenceDedupSignature(
        output_folder="/nonexistent", config=SENT.dt_config(("xxhash", 64), norm, **kw), language=JA
    )
    want = [
        [(h, s) for h, _, s in step.get_hashes(SENT.Document(text=t, id="x"), 0)] for t in texts
    ]
    native = dd._native_sent(SENT.py_config(("xxhash", 64), norm, **kw), JA)
    got = [[tuple(x) for x in doc] for doc in dd._native._sentence_hashes(texts, native)]
    assert got == want


@pytest.fixture(scope="module")
def sd(ref: Reference) -> Any:
    """The sentence-dedup harness (skips without it)."""
    if not SENT.HAVE_DT:
        pytest.skip("needs datatrove 0.10.1 and spaCy")
    return SENT


@examples(100)
@given(
    texts=st.lists(documents(), min_size=1, max_size=4),
    n=st.integers(1, 3),
    norm=st.one_of(st.just(OFF), SENT.norm_configs),
)
def test_sentences_match_datatrove(sd: Any, texts: list[str], n: int, norm: Any) -> None:
    sentence_hashes(texts, n, norm)


@examples(10)
@given(data=st.data())
def test_sentences_of_long_documents_match_datatrove(
    ref: Reference, sd: Any, data: st.DataObject
) -> None:
    doc = data.draw(long_documents(ref.chunk), label="doc")
    sentence_hashes([doc], data.draw(st.integers(1, 3), label="n"), OFF)


JA_SENTENCES = [
    "猫がマットの上で寝ている。", "また雨が降っている！", "元気ですか？",
    "田中さんは東京へ行った。",
    "価格は３．５％上がった。", "「こんにちは」と彼は言った。", "これは本です", "ありがとう！！",
    "えっ？本当？", "第3四半期の売上高は前年同期比12.5%増の1,234億円でした。",
]  # fmt: skip
SENT_SEPS = ["", "", " ", "  ", "\n", "\n\n", "　", " \n", "\r\n"]


@st.composite
def corpora(draw: st.DrawFn) -> list[str]:
    """Small corpora whose documents share sentences."""
    pool = draw(st.lists(st.sampled_from(JA_SENTENCES) | lines(), min_size=1, max_size=8))
    docs = []
    for _ in range(draw(st.integers(1, 6))):
        doc = draw(st.sampled_from(SENT_SEPS))
        for _ in range(draw(st.integers(0, 10))):
            doc += draw(st.sampled_from(pool)) + draw(st.sampled_from(SENT_SEPS))
        docs.append(doc)
    return docs


SD_MINS: list[dict[str, Any]] = [
    {"min_doc_words": 0, "min_num_sentences": 0},
    {"min_doc_words": 5, "min_num_sentences": 2},
    {"min_doc_words": 50, "min_num_sentences": 3, "min_words_to_remove_span": 3},
]


def stages(sd: Any, texts: list[str], kw: dict[str, Any], work: Path) -> None:
    """datatrove's three stages, pyroid's drop-ins (natively) and SentenceDedup agree."""
    import pyroid.integrations.datatrove as pdt

    cfg = sd.dt_config(**kw)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pyroid.FallbackWarning)
        want = sd.run_stages(sd.dt_sd, texts, cfg, work / "dt", language=JA)
        got = sd.run_stages(pdt, texts, cfg, work / "pr", language=JA, batch_size=2)
        text, keep = dd.SentenceDedup(sd.py_config(**kw), language=JA).fit(texts)
    for key in ("sig", "dup", "excl"):
        assert got[key] == want[key], key
    for (gi, gt), (wi, wt) in zip(got["kept"], want["kept"], strict=True):
        assert (gi, len(gt)) == (wi, len(wt)), _first_difference(gt, wt)
        assert gt == wt, _first_difference(gt, wt)
    assert len(got["kept"]) == len(want["kept"])
    kept = dict(want["kept"])
    assert keep == [str(i) in kept for i in range(len(texts))]
    assert [t for t, k in zip(text, keep, strict=True) if k] == [t for _, t in want["kept"]]
    assert dd.SentenceDedup(sd.py_config(**kw), language=JA).fit(texts) == (text, keep)


@examples(80)
@given(texts=corpora(), n=st.integers(1, 3), mins=st.sampled_from(SD_MINS))
def test_sentence_dedup_matches_datatrove(
    sd: Any, texts: list[str], n: int, mins: dict[str, Any], tmp_path_factory: Any
) -> None:
    stages(sd, texts, {"n_sentences": n, **mins}, tmp_path_factory.mktemp("ja"))


@examples(8)
@given(data=st.data())
def test_sentence_dedup_of_long_documents_matches_datatrove(
    ref: Reference, sd: Any, data: st.DataObject, tmp_path_factory: Any
) -> None:
    # The repeated lines are duplicate windows, also within one document; stage 3 cuts them
    # out with span_tokenize's (shifted) spans.
    doc = data.draw(long_documents(ref.chunk), label="doc")
    texts = data.draw(st.sampled_from([[doc], [doc[:3000], doc], [doc, doc[-5000:]]]))
    mins = data.draw(st.sampled_from(SD_MINS))
    stages(
        sd,
        texts,
        {"n_sentences": data.draw(st.integers(1, 3)), **mins},
        tmp_path_factory.mktemp("jl"),
    )


@st.composite
def families(draw: st.DrawFn) -> list[str]:
    """Japanese near-duplicates: inserted, deleted, respaced and re-punctuated pieces."""
    docs: list[str] = []
    for _ in range(draw(st.integers(1, 3))):
        base = draw(documents())
        docs.append(base)
        for _ in range(draw(st.integers(0, 3))):
            i = draw(st.integers(0, len(base)))
            op = draw(st.sampled_from(["insert", "delete", "space", "punct", "upper"]))
            if op == "insert":
                docs.append(base[:i] + draw(pieces()) + base[i:])
            elif op == "delete":
                docs.append(base[:i] + base[min(i + draw(st.integers(1, 8)), len(base)) :])
            elif op == "space":
                docs.append(base[:i] + draw(st.sampled_from(WHITESPACE)) + base[i:])
            elif op == "punct":
                docs.append(base[:i] + draw(st.sampled_from(PUNCT)) + base[i:])
            else:
                docs.append(base.upper())
    return draw(st.permutations(docs))


@pytest.fixture(scope="module")
def dt_minhash(ref: Reference, tmp_path_factory: pytest.TempPathFactory) -> Any:
    return DEDUP.Datatrove(tmp_path_factory.mktemp("datatrove-minhash"))


@examples(80)
@given(texts=families(), hc=st.sampled_from(DEDUP.HASHES), n=st.sampled_from([1, 2, 5]))
def test_minhash_matches_datatrove(
    dt_minhash: Any, texts: list[str], hc: tuple[str, int], n: int
) -> None:
    DEDUP.check(dt_minhash, JA, texts, hc, n)


@examples(8)
@given(data=st.data())
def test_minhash_of_long_documents_matches_datatrove(
    ref: Reference, dt_minhash: Any, data: st.DataObject
) -> None:
    doc = data.draw(long_documents(ref.chunk), label="doc")
    n = data.draw(st.sampled_from([1, 3, 5]), label="n_grams")
    kw: dict[str, Any] = {"n_grams": n, "num_buckets": 6, "hashes_per_bucket": 2}
    texts = [doc, doc[: len(doc) // 2] + "。" + doc[len(doc) // 2 :]]
    want = dt_minhash.signatures(dt_minhash.config("xxhash", 64, **kw), JA, texts)
    assert dd.minhash_signatures(texts, dd.MinhashConfig(**kw), language=JA) == want


def test_python_tokenizer_is_identical(ref: Reference) -> None:
    docs = [*DOCS, at_chunk_ends("今日は晴れ。明日は雨。\n" * 4000, "space", ref.chunk)]
    for cfg in [*(configs(k)[0] for k in KINDS), *chain()]:
        assert pf.apply(docs, [cfg], python_tokenizer=True) == pf.apply(docs, [cfg]), cfg
    for n in (1, 5):
        native = dd.minhash_signatures(docs, language=JA, n_grams=n)
        assert dd.minhash_signatures(docs, language=JA, n_grams=n, python_tokenizer=True) == native
        assert dd.minhash_signatures(docs, language=ref.tok, n_grams=n) == native


def test_fineweb2_documents_match_datatrove(ref: Reference) -> None:
    """The Japanese FineWeb-2 slice: each document's words and sentence spans (through C4)."""
    texts = FILTERS.fineweb2_texts(JA)
    for t in texts:
        same_words(ref, t)
    cfg = configs("c4")[-2]  # keeps every sentence
    check(ref, [cfg], texts)


# ---- SudachiPy's input limit -----------------------------------------------------------

#: SudachiPy normalizes "㍿" to "株式会社" (12 bytes) and refuses an input whose normalized
#: text is over 65,535 bytes: 5,461 of them and "abc" are 65,535, with "abcd" one too many.
AT_LIMIT = "㍿" * 5461 + "abc"
OVER_LIMIT = "㍿" * 5461 + "abcd"
#: The limit applies to each 40,000-byte chunk: 6,000 "㍿" with ASCII between them.
CHUNKED = ("㍿" * 2000 + "a" * 34_000) * 3


@needs_dictionary
def test_sudachi_input_limit_is_a_value_error() -> None:
    for text in (AT_LIMIT, CHUNKED):  # no error
        assert isinstance(pf.gopher_quality_reasons(text, language=JA), str)
        assert dd.minhash_signatures(text, language=JA, norm_config=OFF) is not None
    for f in (pf.gopher_quality, pf.gopher_quality_reasons, pf.gopher_repetition_reasons):
        with pytest.raises(ValueError, match="Input is too long") as e:
            f(["日本語", OVER_LIMIT], language=JA)
        assert "index 1" in _notes(e.value), _notes(e.value)
        with pytest.raises(ValueError, match="Input is too long") as e:
            f(OVER_LIMIT, language=JA)
        assert "the str" in _notes(e.value), _notes(e.value)
    with pytest.raises(ValueError, match="Input is too long"):
        pf.apply([OVER_LIMIT + "\n" + AT_LIMIT], [pf.C4Quality(language=JA, split_paragraph=False)])
    with pytest.raises(ValueError, match="Input is too long"):
        dd.minhash_signatures(["日本語", OVER_LIMIT], language=JA, norm_config=OFF)
    with pytest.raises(ValueError, match="Input is too long"):
        dd.SentenceDedup(language=JA).add(["日本語。", OVER_LIMIT])
    # C4 splits each line into sentences: one at a time, they stay under the limit.
    lines_ = OVER_LIMIT[:1000] + "\n" + OVER_LIMIT[1000:]
    per_line: dict[str, Any] = {
        "filter_no_terminal_punct": False,
        "min_words_per_line": -1,
        "min_num_sentences": 1,
    }
    assert pf.c4_quality_reasons(lines_, language=JA, **per_line) is None
    with pytest.raises(ValueError, match="Input is too long"):
        pf.c4_quality_reasons(lines_, language=JA, split_paragraph=False, **per_line)


def test_sudachi_input_limit_matches_datatrove(ref: Reference) -> None:
    for text in (AT_LIMIT, CHUNKED, "日本語。" + AT_LIMIT[4:], "a" * 65_536):
        same_words(ref, text)
    for text in (OVER_LIMIT, "日本語。" * 3 + OVER_LIMIT, CHUNKED.replace("a" * 34_000, "a" * 9)):
        for kind in KINDS:
            check(ref, [configs(kind)[0]], [text])
    with pytest.raises(Exception, match="Input is too long") as e:
        ref.tok.word_tokenize(OVER_LIMIT)
    assert type(e.value).__name__ == "SudachiError"


def test_is_alpha_retry_after_the_input_limit(ref: Reference) -> None:
    # SpaCyTokenizer._do_tokenize: on any exception, a text with "IS_ALPHA" in it is tokenized
    # again with every "IS_ALPHA" removed. Here that brings the input under SudachiPy's limit.
    text = "IS_ALPHA" + AT_LIMIT[:-3]
    assert len(ref.tok.word_tokenize(text)) == 5461
    same_words(ref, text)


# ---- containers, threads, errors (no datatrove needed) -------------------------------------


def sample_docs(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        ls = []
        for _ in range(rng.randint(0, 10)):
            ps = (rng.choice(PIECES) + rng.choice(["", "", "", " ", "　"]) for _ in range(30))
            ls.append("".join(ps)[: rng.randint(0, 200)] + rng.choice(ENDINGS))
        out.append(rng.choice(["\n", "\n\n", " "]).join(ls))
    return out


@needs_dictionary
def test_container_kinds() -> None:
    steps = chain()
    res = pf.apply(DOCS, steps)
    assert isinstance(res.keep, list)
    assert pf.apply(tuple(DOCS), steps) == tuple(map(tuple, res))
    for i, d in enumerate(DOCS):
        one = pf.apply(d, steps)
        assert one == (res.keep[i], res.reason[i], res.text[i])
        if one.text == d:
            assert one.text is d  # unchanged texts are the original objects
    for f in ALL_FUNCS:
        many = f(DOCS, language=JA)
        assert f(tuple(DOCS), language=JA) == tuple(many)
        assert [f(d, language=JA) for d in DOCS] == many
    sigs = dd.minhash_signatures(DOCS, language=JA)
    assert dd.minhash_signatures(tuple(DOCS), language=JA) == sigs
    assert [dd.minhash_signatures(d, language=JA) for d in DOCS] == sigs
    assert sigs[1] is None  # no shingles
    keep = dd.minhash_dedup([*DOCS, DOCS[0]], language=JA)
    assert keep == [True] * len(DOCS) + [False]
    assert dd.minhash_dedup(tuple(DOCS), language=JA) == (True,) * len(DOCS)
    sd = dd.sentence_dedup([*DOCS, DOCS[5]], language=JA, **SENT.NO_MIN)
    assert sd.keep[-1] is False
    assert sd.text[3] == "ラーメンを食べた。"  # its own repeats too
    tup = dd.sentence_dedup((*DOCS, DOCS[5]), language=JA, **SENT.NO_MIN)
    assert tup == (tuple(sd.text), tuple(sd.keep))
    assert tup.text[0] is DOCS[0]
    assert dd.sentence_dedup(DOCS[0], language=JA, **SENT.NO_MIN) == (DOCS[0], True)


@needs_dictionary
@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("chunks", [1, 3])
def test_arrow_without_pyarrow(large: bool, chunks: int) -> None:
    values: list[str | None] = [*DOCS[:3], None, *DOCS[3:], None]
    col = _testing.column(values, large=large, chunks=chunks, name="text")
    plain = [v for v in values if v is not None]

    def spread(xs: Sequence[Any]) -> list[Any]:
        it = iter(xs)
        return [None if v is None else next(it) for v in values]

    want = pf.apply(plain, chain())
    res = pf.apply(col, chain())
    assert res.keep.to_pylist() == spread(want.keep)
    assert res.reason.to_pylist() == spread(want.reason)
    assert res.text.to_pylist() == spread(want.text)
    assert res.text.type == ("large_string" if large else "string")
    assert res.keep.name == "text"
    for f in ALL_FUNCS:
        assert f(col, language=JA).to_pylist() == spread(f(plain, language=JA))
    assert dd.minhash_signatures(col, language=JA) == spread(
        dd.minhash_signatures(plain, language=JA)
    )
    mask = dd.minhash_dedup(col, language=JA)
    assert mask.to_pylist() == spread(dd.minhash_dedup(plain, language=JA))
    text, keep = dd.sentence_dedup(col, language=JA, **SENT.NO_MIN)
    want_sd = dd.sentence_dedup(plain, language=JA, **SENT.NO_MIN)
    assert text.to_pylist() == spread(want_sd.text)
    assert keep.to_pylist() == spread(want_sd.keep)


@needs_dictionary
def test_pyarrow(pa: Any) -> None:
    arr = pa.chunked_array([DOCS[:4], [None, *DOCS[4:]]], type=pa.string_view())
    want = pf.apply(DOCS, chain())
    res = pf.apply(arr, chain())
    assert res.keep.to_pylist() == [*want.keep[:4], None, *want.keep[4:]]
    assert pa.chunked_array(res.text).to_pylist() == [*want.text[:4], None, *want.text[4:]]
    sliced = pa.array(DOCS, type=pa.large_string()).slice(2)
    got = pf.gopher_repetition_reasons(sliced, language=JA)
    assert got.to_pylist() == pf.gopher_repetition_reasons(DOCS[2:], language=JA)
    assert dd.minhash_signatures(sliced, language=JA) == dd.minhash_signatures(
        DOCS[2:], language=JA
    )


@needs_dictionary
@pytest.mark.parametrize("threads", [None, 1, 4])
def test_threads(threads: int | None) -> None:
    docs = sample_docs(300, seed=1)
    assert pf.apply(docs, chain(), threads=threads) == pf.apply(docs, chain(), threads=1)
    for f in (pf.gopher_quality_reasons, pf.c4_lines):
        assert f(docs, language=JA, threads=threads) == f(docs, language=JA, threads=1)
    want = dd.minhash_signatures(docs, language=JA, threads=1)
    assert dd.minhash_signatures(docs, language=JA, threads=threads) == want
    d = dd.MinHashDedup(language=JA, threads=threads)
    d.add(docs[:150])
    d.add(docs[150:] + docs[:20])
    assert d.keep_mask() == dd.minhash_dedup(docs + docs[:20], language=JA, threads=1)
    sd = dd.SentenceDedup(language=JA, threads=threads, **SENT.NO_MIN).fit(docs)
    assert sd == dd.SentenceDedup(language=JA, threads=1, **SENT.NO_MIN).fit(docs)


@needs_dictionary
@pytest.mark.timeout(300)
def test_2000_documents() -> None:
    docs = sample_docs(2000, seed=2)
    res = pf.apply(docs, chain())
    assert len(res.keep) == 2000
    assert len(set(res.reason)) > 3
    sigs = dd.minhash_signatures(docs, language=JA)
    assert len(sigs) == 2000
    # 100 copies at the end: removed (those with shingles); the others as without them.
    keep = dd.minhash_dedup(docs + docs[:100], language=JA)
    assert keep[:2000] == dd.minhash_dedup(docs, language=JA)
    assert not any(k for k, s in zip(keep[2000:], sigs, strict=False) if s is not None)
    text, kept = dd.sentence_dedup(docs + docs[:100], language=JA)
    assert (text[:2000], kept[:2000]) == dd.sentence_dedup(docs, language=JA)
    assert sum(t != d for t, d in zip(text[2000:], docs, strict=False)) > 50


@needs_dictionary
def test_input_errors() -> None:
    with pytest.raises(pyroid.InputError, match="index 1"):
        pf.gopher_quality(["日本語", 3], language=JA)  # type: ignore[list-item]
    with pytest.raises(pyroid.InputError):
        pf.apply(b"bytes", chain())  # type: ignore[arg-type]
    with pytest.raises(pyroid.InputError):
        dd.minhash_signatures(["日本語", None], language=JA)  # type: ignore[list-item]
    with pytest.raises(pyroid.InputError):
        dd.SentenceDedup(language=JA).add([1])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="threads"):
        pf.apply(DOCS * 10, chain(), threads=0)


@needs_dictionary
@examples(30)
@given(doc=documents(), pos=st.integers(0, 10_000), cs=st.characters(categories=["Cs"]))
def test_lone_surrogates_raise_unicode_encode_error(doc: str, pos: int, cs: str) -> None:
    at = pos % (len(doc) + 1)
    bad = doc[:at] + cs + doc[at:]
    for f in ALL_FUNCS:
        with pytest.raises(UnicodeEncodeError) as e:
            f(["日本語", bad], language=JA)
        assert "index 1" in _notes(e.value), _notes(e.value)
    with pytest.raises(UnicodeEncodeError):
        pf.apply(bad, chain())
    with pytest.raises(UnicodeEncodeError):
        dd.minhash_signatures(["日本語", bad], language=JA)
    with pytest.raises(UnicodeEncodeError):
        dd.MinHashDedup(language=JA).add([bad])
    with pytest.raises(UnicodeEncodeError):
        dd.SentenceDedup(language=JA).add([bad])


# ---- the dictionary, in child interpreters ------------------------------------------------


def system_dic() -> Path:
    """sudachidict_core's system.dic (skip without it)."""
    spec = importlib.util.find_spec("sudachidict_core")
    if spec is None or spec.origin is None:
        pytest.skip("sudachidict-core is not installed")
    return Path(spec.origin).parent / "resources" / "system.dic"


CALLS = """
import os, pyroid, pyroid.filters as pf, pyroid.dedup as dd
calls = [
    ("jpn_Jpan", lambda: pf.gopher_quality(["日本語"], language="jpn_Jpan")),
    ("ja", lambda: pf.apply(["日本語"], [pf.C4Quality(language="ja")])),
    ("jpn", lambda: dd.minhash_signatures(["日本語"], language="jpn")),
    ("ja_Jpan", lambda: dd.MinHashDedup(language="ja_Jpan")),
    ("jpn_Jpan", lambda: dd.SentenceDedup(language="jpn_Jpan")),
]
"""

BAD_DICTIONARY = (
    CALLS
    + """
assert pf.languages()["jpn_Jpan"].native  # a file to load: native until it fails to load
for path, want in CASES:
    os.environ["PYROID_SUDACHI_DICT"] = path
    for _, call in calls * 2:  # every use tries again
        try:
            call()
        except Exception as e:
            assert type(e).__name__ == want, (path, type(e), e)
            assert path in str(e), (path, e)
            msg = str(e)
        else:
            raise SystemExit(f"{path}: no {want}")
    print(want, msg)
# Other languages do not need the dictionary.
assert pf.gopher_quality_reasons("Ein kurzer Satz.", language="deu_Latn") == "gopher_short_doc"
"""
)


@pytest.mark.thread_unsafe
def test_missing_or_invalid_dictionary(run_python: Callable[..., Any], tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.dic"
    garbage.write_bytes(b"not a dictionary\n" * 100)
    empty = tmp_path / "empty.dic"
    empty.write_bytes(b"")
    old = tmp_path / "old.dic"  # the header of a SudachiPy 0.6 system dictionary
    old.write_bytes(struct.pack("<Q", 0x7366_D3F1_8BD1_11E7) + bytes(4096))
    cases = [
        (str(tmp_path / "missing" / "system.dic"), "OSError"),
        (str(tmp_path), "OSError"),  # a directory
        (str(garbage), "ValueError"),
        (str(empty), "ValueError"),
        (str(old), "ValueError"),
    ]
    if DICTIONARY and "PYROID_SUDACHI_DICT" not in os.environ:
        data = system_dic().read_bytes()[: 1 << 20]
        for size in (8, 100, 4096, 1 << 20):  # truncated dictionaries
            cut = tmp_path / f"cut{size}.dic"
            cut.write_bytes(data[:size])
            cases.append((str(cut), "ValueError"))
    code = f"CASES = {cases!r}\n" + BAD_DICTIONARY
    r = run_python(code, env={"PYROID_SUDACHI_DICT": cases[0][0]})
    assert r.returncode == 0, r.stderr
    out = r.stdout.splitlines()
    assert len(out) == len(cases), out
    assert "cannot read the Sudachi dictionary" in out[0]
    assert "SudachiPy 0.6 or older" in out[4], out[4]


NO_DICTIONARY = (
    """
import sys
sys.modules["sudachidict_core"] = None  # not installed
"""
    + CALLS
    + """
import importlib.util
langs = pf.languages()
assert not any(langs[c].native for c in ("ja", "ja_Jpan", "jpn", "jpn_Jpan")), langs["jpn_Jpan"]
assert sum(v.native for v in langs.values()) == 4009
for code, call in calls:
    try:
        call()
    except NotImplementedError as e:
        print("NotImplementedError:", code, str(e).replace("\\n", " "))
    else:
        raise SystemExit("no NotImplementedError")
d = dd.SentenceDedup(
    language="jpn_Jpan", split_sentences=False, min_doc_words=0, min_num_sentences=0
)
assert d.fit(["a\\nb\\nc", "a\\nb\\nc"]) == (["a\\nb\\nc", ""], [True, False])
if all(importlib.util.find_spec(m) for m in ("datatrove", "spacy", "sudachipy")):
    try:
        pf.gopher_quality(["日本語"], language="jpn_Jpan", python_tokenizer=True)
    except pyroid.PyroidError as e:  # SudachiPy has no dictionary either
        print("PyroidError:", e)
    else:
        raise SystemExit("python_tokenizer=True found a dictionary")
"""
)


@pytest.mark.thread_unsafe
def test_no_dictionary(run_python: Callable[..., Any]) -> None:
    r = run_python(NO_DICTIONARY, env={"PYROID_SUDACHI_DICT": None})
    assert r.returncode == 0, r.stderr
    out = r.stdout.splitlines()
    errors = [ln.split(" ", 2)[1:] for ln in out if ln.startswith("NotImplementedError:")]
    assert len(errors) == 5, out
    for code, msg in errors[:4]:  # the filters and MinHash
        for part in (f"'{code}'", "sudachidict-core", "python_tokenizer=True"):
            assert part in msg, (part, msg)
    code, msg = errors[4]  # SentenceDedup: datatrove's steps, or lines
    assert f"'{code}'" in msg
    assert "SudachiPy" in msg
    for ln in out:
        if ln.startswith("PyroidError:"):
            assert "sudachidict-core" in ln, ln


def _steps_code(steps: Sequence[Any]) -> str:
    return "[" + ", ".join(repr(s) for s in steps) + "]"


ELSEWHERE = """
import json, os, sys
from pyroid.filters import C4Quality, FineWebQuality, GopherQuality, GopherRepetition
import pyroid.filters as pf, pyroid.dedup as dd
docs = $docs
steps = $steps
native = pf.languages()["jpn_Jpan"].native
assert native is $native, native
python = not native
res = pf.apply(docs, steps, python_tokenizer=python)
sigs = dd.minhash_signatures(docs, language="jpn_Jpan", n_grams=2, python_tokenizer=python)
if native:
    # Loaded once per process: the variable is not read again.
    os.environ["PYROID_SUDACHI_DICT"] = "/nonexistent/system.dic"
    assert pf.apply(docs, steps) == res
    assert dd.SentenceDedup(language="ja").fit(docs).keep
else:
    try:
        pf.apply(docs, steps)
    except NotImplementedError:
        pass
    else:
        raise SystemExit("native without a dictionary")
print(json.dumps([list(res.keep), list(res.reason), list(res.text), sigs]))
"""

#: pyroid finds no dictionary, SudachiPy (and so datatrove) does.
HIDDEN_FROM_PYROID = """
import importlib.util
import sudachipy  # binds importlib.util.find_spec before it is replaced
_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, package=None: (
    None if name == "sudachidict_core" else _find_spec(name, package)
)
"""
#: Only $PYROID_SUDACHI_DICT.
ONLY_THE_VARIABLE = """
import sys
sys.modules["sudachidict_core"] = None
"""


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("where", ["variable", "sudachipy-only"])
def test_dictionary_found_elsewhere(run_python: Callable[..., Any], where: str) -> None:
    """``$PYROID_SUDACHI_DICT``, or ``python_tokenizer=True`` with a dictionary only SudachiPy
    finds: the results of the native path with sudachidict_core."""
    if not DICTIONARY:
        pytest.skip("needs a SudachiDict for the native results")
    if where == "sudachipy-only":
        for m in ("datatrove", "spacy", "sudachipy"):
            pytest.importorskip(m)
    docs = [*DOCS, "日本語の文です。" * 3000]
    steps = chain()
    code = (
        HIDDEN_FROM_PYROID if where == "sudachipy-only" else ONLY_THE_VARIABLE
    ) + string.Template(ELSEWHERE).substitute(
        docs=repr(docs), steps=_steps_code(steps), native=where == "variable"
    )
    env = {"PYROID_SUDACHI_DICT": str(system_dic()) if where == "variable" else None}
    r = run_python(code, env=env, timeout=300)
    assert r.returncode == 0, r.stderr
    res = pf.apply(docs, steps)
    sigs = dd.minhash_signatures(docs, language=JA, n_grams=2)
    assert json.loads(r.stdout) == [list(res.keep), list(res.reason), list(res.text), sigs]


FROM_THREADS = """
import json, threading
import pyroid.filters as pf, pyroid.dedup as dd
docs = $docs
barrier = threading.Barrier(8)
out, errors = [], []

def work(i):
    try:
        barrier.wait()  # the first use, from 8 threads at once
        if i % 2:
            out.append(("f", pf.gopher_quality_reasons(docs, language="jpn_Jpan")))
        else:
            out.append(("m", dd.minhash_signatures(docs, language="ja", n_grams=2)))
    except BaseException as e:
        errors.append(repr(e))

ts = [threading.Thread(target=work, args=(i,)) for i in range(8)]
for t in ts:
    t.start()
for t in ts:
    t.join(timeout=240)
assert not errors, errors
assert len(out) == 8
f = {json.dumps(v) for k, v in out if k == "f"}
m = {json.dumps(v) for k, v in out if k == "m"}
assert len(f) == len(m) == 1, (len(f), len(m))
print(json.dumps([json.loads(f.pop()), json.loads(m.pop())]))
"""


@pytest.mark.thread_unsafe
@needs_dictionary
def test_first_use_from_threads(run_python: Callable[..., Any]) -> None:
    docs = sample_docs(40, seed=3)
    code = string.Template(FROM_THREADS).substitute(docs=repr(docs))
    r = run_python(code, timeout=300)
    assert r.returncode == 0, r.stderr
    want = [
        pf.gopher_quality_reasons(docs, language=JA),
        dd.minhash_signatures(docs, language=JA, n_grams=2),
    ]
    assert json.loads(r.stdout) == want
