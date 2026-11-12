"""Vietnamese and Thai in pyroid.filters and pyroid.dedup: datatrove's ``SpaCyTokenizer("vi")``
and ``ThaiTokenizer``.

datatrove tokenizes 16 codes (``vie_Latn`` among them) with spaCy's Vietnamese tokenizer on pyvi
0.1.1, and 20 codes (``tha_Thai`` among them) with PyThaiNLP 5.4.0: newmm-safe words and crfcut
sentences. pyroid ships pyvi's model and word list, so Vietnamese is always native. It reads
PyThaiNLP's word list and sentence model from the installed ``pythainlp`` package (or the directory
``$PYROID_PYTHAINLP_CORPUS`` names), so Thai is native only when it finds them; without them the
Thai codes need ``python_tokenizer=True`` like the other non-native codes.

- The language table (no datatrove needed).
- Live datatrove (skips without datatrove and spaCy, and pyvi or PyThaiNLP with python-crfsuite):
  Hypothesis documents from pyvi's and PyThaiNLP's word lists with abbreviations, numbers, URLs,
  e-mails, decomposed accents, Thai signs and every kind of whitespace, short and of several kB
  (Thai runs past newmm's 140-character safe-mode window), through every filter and FineWeb-2's
  chains, word lists, sentence splits, MinHash and the sentence-dedup stages; the FineWeb-2
  slices (``$PYROID_DATA/fineweb2/<lang>.parquet``, default ``/data``).
- Container kinds, Arrow, ``threads=``, a 2,000-document batch, input errors, lone surrogates.
- PyThaiNLP in child interpreters: not installed, ``$PYROID_PYTHAINLP_CORPUS`` naming a missing
  or bad directory or the real one, the first use from several threads.

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
import shutil
import string
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
    # datatrove's, spaCy's and the segmenters' tokenizers are not thread-safe; the warning is
    # once per process.
    pytest.mark.thread_unsafe(reason="datatrove, spaCy, pyvi, PyThaiNLP and the warning"),
    # Which path ran, not what the result is (MODULES.md); parity suites ignore it.
    pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning"),
]

_HERE = Path(__file__).parent


def _sibling(name: str) -> Any:
    """A sibling test module, imported by path (the suite runs without python/tests on sys.path)."""
    spec = importlib.util.spec_from_file_location(f"_pyroid_vithai_{name}", _HERE / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: The filters' harness (Datatrove, candidate, configs, CHAINS, FINEWEB2, fineweb2_steps),
#: MinHash's (Datatrove, check, HASHES) and the sentence dedup's (run_stages, dt_config, ...).
FILTERS = _sibling("test_filters_multilingual")
DEDUP = _sibling("test_dedup_multilingual")
SENT = _sibling("test_sentence_dedup_more")

VI, TH = "vie_Latn", "tha_Thai"
LANGS = [VI, TH]
VI_CODES = [
    "bdq", "bdq_Latn", "bru", "bru_Latn", "caq", "caq_Latn", "cmo", "cmo_Latn", "kha", "kha_Latn",
    "vi", "vi_Latn", "vie", "vie_Latn", "wbm", "wbm_Latn",
]  # fmt: skip
TH_CODES = [
    "bzi", "bzi_Thai", "kxm", "kxm_Thai", "lcp", "lcp_Thai", "ms", "ms_Thai", "msa", "msa_Thai",
    "nod", "nod_Thai", "pww", "pww_Thai", "th", "th_Thai", "tha", "tha_Thai", "urk", "urk_Thai",
]  # fmt: skip
STOPS: dict[str, list[str]] = {lang: FILTERS.FINEWEB2[lang]["stopwords"] for lang in LANGS}

#: Whether pyroid finds PyThaiNLP's corpus (Thai is then native), and a SudachiDict (Japanese).
THAI: bool = FILTERS.pythainlp_corpus()
SUDACHI: bool = FILTERS.sudachi_dictionary()
needs_thai = pytest.mark.skipif(
    not THAI, reason="no PyThaiNLP (pip install pythainlp): Thai is not native"
)
#: The languages, for tests that need no datatrove (Thai only when it is native).
NATIVE_LANGS = [VI, pytest.param(TH, marks=needs_thai)]
#: What datatrove's tokenizer needs besides datatrove and spaCy: pyvi (its CRF model loads with
#: sklearn-crfsuite) and PyThaiNLP (whose crfcut sentences need python-crfsuite).
MODULES: dict[str, tuple[str, ...]] = {
    VI: ("pyvi", "sklearn_crfsuite"),
    TH: ("pythainlp", "pycrfsuite"),
}


def examples(n: int) -> settings:
    """``n`` examples per test, five times as many under the ``ci`` profile (300 examples)."""
    env = os.environ.get("PYROID_FILTERS_EXAMPLES")
    profile = settings.default  # the loaded profile (conftest.py)
    scale = max((profile.max_examples if profile is not None else 60) // 60, 1)
    return settings(
        max_examples=int(env) if env else n * scale,
        deadline=None,
        suppress_health_check=list(HealthCheck),
    )


def _notes(exc: BaseException) -> str:
    return "\n".join(getattr(exc, "__notes__", []))


def _event(value: str) -> None:
    with contextlib.suppress(InvalidArgument):  # outside a Hypothesis test
        event(value)


# ---- word lists ----------------------------------------------------------------------


def _package_file(package: str, *parts: str) -> Path | None:
    """A data file of an installed package, found without importing it."""
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin).parent.joinpath(*parts)
    return path if path.is_file() else None


def _word_list(paths: Sequence[Path | None], fallback: list[str]) -> list[str]:
    """The non-empty lines of the first file found, else ``fallback``."""
    for path in paths:
        if path is not None and path.is_file():
            words = [w for w in path.read_text(encoding="utf-8").splitlines() if w.strip()]
            if words:
                return words
    return fallback


#: pyvi's words.txt (the installed package's, else pyroid's copy in the source tree).
VI_WORDS = _word_list(
    [
        _package_file("pyvi", "models", "words.txt"),
        _HERE.parents[1] / "crates" / "pyroid-core" / "src" / "pyvi" / "data" / "words.txt",
    ],
    [
        "hà nội", "việt nam", "thủ đô", "của", "là", "và", "những", "người", "thành phố",
        "hồ chí minh", "năm", "đã", "được", "tiếng việt", "học sinh", "trường học", "bóng đá",
        "cà phê", "điện thoại", "máy tính", "một", "trong", "các", "có", "loài", "này",
    ],
)  # fmt: skip
VI_SYLLABLES = sorted({s for w in VI_WORDS for s in w.split()})
#: PyThaiNLP's words_th.txt (the word list of newmm).
TH_WORDS = _word_list(
    [_package_file("pythainlp", "corpus", "words_th.txt")],
    [
        "ภาษาไทย", "ฉัน", "รัก", "มาก", "กรุงเทพมหานคร", "และ", "ที่", "ของ", "ใน", "เป็น", "มี",
        "การ", "ได้", "วันนี้", "อากาศ", "ดี", "ไป", "เดินเล่น", "สวนสาธารณะ", "ประเทศไทย",
        "โรงเรียน", "นักเรียน", "คอมพิวเตอร์", "อาหาร", "เมือง", "หลวง", "ประชากร", "ประมาณ",
    ],
)  # fmt: skip

# ---- Vietnamese documents --------------------------------------------------------------

#: pyvi's abbreviation patterns ([A-ZĐ]+\., Tp., Mr., Mrs., Ms., Dr., ThS.) and near misses.
VI_ABBREV = [
    "TP.", "Tp.", "Mr.", "Mrs.", "Ms.", "Dr.", "ThS.", "TS.", "PGS.", "GS.", "UBND", "TP.HCM",
    "Đ.", "v.v.", "St.", "A.", "ĐH.", "tp.", "mr.", "Tp", "ThS", "TP. HCM", "U.S.A.", "Ths.",
    "ĐHQG.", "Mr.Nam", "ÂU.", "Ư.",
]  # fmt: skip
#: pyvi's specials (==>, ->, ..., >>) and their neighbours.
VI_SPECIALS = ["==>", "->", "...", ">>", "-->", "=>", "<<", "....", ". . .", "->>", "==>>", "…"]
#: \d+([\.,_]\d+)+ and what it does not match.
VI_NUMBERS = [
    "1.000", "1.000.000", "3,14", "12_34", "1.2.3", "1..2", "2024", "1,", ",5", "10%",
    "25/12/2024", "12-3", "08:30", "１２３", "١٢٣", "²", "½", "1.000,50đ", "100.000 VNĐ", "1_000",
    "0,5", "3.", "x2", "2x", "1.000.000,00", "1,2,3", "_1", "1_", "௧௨", "1.a", "1.000đ",
]  # fmt: skip
#: \w+://[^\s]+ and the e-mail pattern.
VI_WEB = [
    "https://vnexpress.net/tin-tuc?id=1", "http://a.b/c.", "ftp://x.y", "www.baomoi.com",
    "info@vnexpress.net", "a.b+c@d-e.vn", "x@y", "user@mail.", "mailto:a@b.vn",
    "tên.miền@công-ty.vn", "https://vi.wikipedia.org/wiki/Việt_Nam", "a://", "://x", "@user",
    "#hashtag", "https://a.vn/x?y=1,", "Nam@",
]  # fmt: skip
#: Case and \w oddities (Python's str methods feed pyvi's CRF features), other scripts. No
#: IS_ALPHA token: datatrove's words for it depend on what spaCy saw before (filters.pyi).
VI_OTHER = [
    "COVID-19", "iPhone", "Wi-Fi", "e-mail", "Đ", "đ", "ĐẠI HỌC", "Ư", "ơ", "ǅ", "İ", "ß", "ΣΑΣ",
    "ﬁ", "Ⅻ", "a_b", "_", "__init__", "😀", "👍🏽", "IS_DIGIT", "Hello", "the", "Москва", "日本",
    "ภาษา", "ŉ", "ǰ", "ᾈ", "Ǆ", "ǲ", "µ", "ſ",
]  # fmt: skip
#: Decomposed accents (pyvi's spaCy tokenizer does not normalize): marks are [^\w\s].
VI_NFD = [
    "Vie\u0323\u0302t", "e\u0301", "a\u0300", "o\u031b", "u\u031b\u0300", "\u0301",
    "Ha\u0300 No\u0323\u0302i", "A\u0306", "Đi\u0323a", "\u0303\u0303",
]  # fmt: skip
VI_PUNCT = [
    ".", ",", "!", "?", ";", ":", "(", ")", '"', "“", "”", "'", "‘", "’", "…", "-", "–", "—", "/",
    "[", "]", "«", "»", "?!", "!!", "%", "&", "*", "+", "=", "。", "¿",
]  # fmt: skip
VI_WS = [" "] * 12 + [
    "  ", "\t", "\xa0", "\u3000", "\u2009", "\u200b", "\x0c", "\r", " \t ", "\u202f",
]  # fmt: skip
VI_SEPS = [" "] * 16 + [""] * 4 + VI_WS
VI_ENDINGS = [".", ".", ".", "!", "?", "", "...", "…", ":", '."', "?!", ". ", "!!!", ";"]
_VI_CHARS = "".join(sorted({c for w in VI_SYLLABLES[::7] for c in w} | set(".,!?:;-()")))


def _vi_words() -> st.SearchStrategy[str]:
    words = st.sampled_from(VI_WORDS)
    return st.one_of(
        words,
        words,
        words,
        words.map(str.title),
        words.map(str.upper),
        words.map(str.capitalize),
        st.sampled_from(VI_SYLLABLES),
    )


def vi_pieces() -> st.SearchStrategy[str]:
    return st.one_of(
        _vi_words(),
        _vi_words(),
        _vi_words(),
        _vi_words(),
        st.sampled_from(VI_ABBREV),
        st.sampled_from(VI_SPECIALS),
        st.sampled_from(VI_NUMBERS),
        st.sampled_from(VI_WEB),
        st.sampled_from(VI_OTHER),
        st.sampled_from(VI_NFD),
        st.sampled_from(VI_PUNCT),
        st.sampled_from(VI_PUNCT),
        st.text(st.sampled_from(_VI_CHARS), min_size=1, max_size=6),
        st.text(st.characters(exclude_categories=["Cs"]), min_size=1, max_size=3),
    )


# ---- Thai documents ------------------------------------------------------------------

#: Signs, vowels and tone marks alone and in clusters (newmm's TCC rules), Thai punctuation.
TH_SIGNS = [
    "ๆ", "ฯ", "ฯลฯ", "๏", "๚", "๛", "฿", "์", "ํ", "๎", "็", "่", "้", "๊", "๋", "ั", "ะ", "า",
    "ำ", "เ", "แ", "โ", "ใ", "ไ", "ๅ", "ฤๅ", "ฦๅ", "ก์", "ก็", "เกาะ", "แม่", "น้ำ", "กำ", "ฤ",
    "เเ", "ไๆ", "ๆๆ", "ั่", "เก็", "กุ้ง", "ปั๊ม", "เปล่า",
]  # fmt: skip
#: Numbers PyThaiNLP rejoins (":", ",", "." between digits) or not, Thai digits.
TH_NUMBERS = [
    "1,000", "1.5", "12:30", "3.14", "1,000.50", "๑,๐๐๐", "๑๒๓", "๒๕๖๗", "10%", "2567", "1.",
    ".5", "12:00น", "192.168.1.1", "10:30:45", "1:", ":1", "1,,2", "๑๒:๓๐", "1.2.3", "0",
    "1,234,567", "12.", "๓.๑๔", "1:2:3:", "٣٤", "１２", "12,", "1 000",
]  # fmt: skip
#: Latin runs ([-a-zA-Z]+), other non-Thai runs, scripts.
TH_LATIN = [
    "abc", "Bangkok", "COVID-19", "iPhone", "e-mail", "-", "a-b-c", "---", "Hello World", "ABC",
    "x", "Ünïcödé", "Москва", "日本語", "😀", "TV", "OK", "a1b2", "_x_", "Việt",
]  # fmt: skip
TH_WEB = ["https://www.example.co.th/a?b=1", "info@example.co.th", "www.sanook.com", "#แท็ก"]
TH_PUNCT = [
    "!", ".", "?", "!!", "...", "?!", "“", "”", "(", ")", '"', "'", ",", ":", ";", "-", "/", "…",
    "!.", ".?", "。",
]  # fmt: skip
#: Thai separates phrases, not words: mostly nothing between pieces.
TH_SEPS = [""] * 14 + [" ", " ", " ", "  ", "\t", "\xa0", "\u3000", "\u200b", "\r", "\u2009"]
TH_WS = [" ", " ", "  ", "\t", "\xa0", "\u3000", "\u200b", "\u2009"]
TH_ENDINGS = ["", "", " ", ".", "!", "?", "ฯ", "ๆ", "...", "!!", "?!", ". ", "  "]
_THAI_BLOCK = st.characters(min_codepoint=0x0E00, max_codepoint=0x0E7F)


def th_pieces() -> st.SearchStrategy[str]:
    words = st.sampled_from(TH_WORDS)
    return st.one_of(
        words,
        words,
        words,
        words,
        words,
        st.lists(words, min_size=2, max_size=5).map("".join),
        st.sampled_from(TH_SIGNS),
        st.sampled_from(TH_NUMBERS),
        st.sampled_from(TH_LATIN),
        st.sampled_from(TH_WEB),
        st.sampled_from(TH_PUNCT),
        st.text(_THAI_BLOCK, min_size=1, max_size=8),
        st.text(st.characters(exclude_categories=["Cs"]), min_size=1, max_size=3),
    )


# ---- documents of either language ---------------------------------------------------------

LINE_SEPS = ["\n"] * 8 + ["\n\n"] * 3 + ["\r\n", "\r", "\n \n", "\x85", "\u2028", "\n\u3000"]
PIECES: dict[str, Callable[[], st.SearchStrategy[str]]] = {VI: vi_pieces, TH: th_pieces}
SEPS = {VI: VI_SEPS, TH: TH_SEPS}
WS = {VI: VI_WS, TH: TH_WS}
ENDINGS = {VI: VI_ENDINGS, TH: TH_ENDINGS}
SPECIAL_DOCS = ["", " ", "\n", "　", ".", "!", " \n ", "\u200b", "ๆ", "\u0301"]


@st.composite
def lines(draw: st.DrawFn, lang: str) -> str:
    """Pieces of ``lang`` with its separators between them, and an ending."""
    ps = draw(st.lists(PIECES[lang](), max_size=24))
    seps = draw(st.lists(st.sampled_from(SEPS[lang]), min_size=len(ps), max_size=len(ps)))
    line = "".join(s + p for s, p in zip(seps, ps, strict=True))[len(seps[0]) if seps else 0 :]
    line += draw(st.sampled_from(ENDINGS[lang]))
    if draw(st.integers(0, 7)) == 0:
        line = draw(st.sampled_from(WS[lang])) + line
    return line


@st.composite
def documents(draw: st.DrawFn, lang: str) -> str:
    """0 to ~16 lines, sometimes repeated (for the repetition checks)."""
    shape = draw(st.integers(0, 9))
    if shape == 0:
        return draw(st.sampled_from(SPECIAL_DOCS))
    if shape == 1:  # one phrase over and over
        phrase = draw(lines(lang))
        sep = draw(st.sampled_from(["", " ", "\n", ". ", ".\n", "\n\n", "! "]))
        return sep.join([phrase] * draw(st.integers(1, 30)))
    ls = draw(st.lists(lines(lang), min_size=1, max_size=16))
    picks = list(range(len(ls)))
    for _ in range(draw(st.sampled_from([0, 0, 1, 3, 8]))):
        picks.insert(draw(st.integers(0, len(picks))), draw(st.integers(0, len(ls) - 1)))
    seps = draw(st.lists(st.sampled_from(LINE_SEPS), min_size=len(picks), max_size=len(picks)))
    doc = "".join(ls[i] + s for i, s in zip(picks, seps, strict=True))
    return doc.rstrip("\n") if draw(st.booleans()) else doc


#: The pools ``random`` draws long documents and sample batches from.
RNG_POOLS: dict[str, list[list[str]]] = {
    VI: [VI_WORDS, VI_WORDS, VI_WORDS, VI_SYLLABLES, VI_ABBREV, VI_NUMBERS, VI_PUNCT, VI_WEB],
    TH: [TH_WORDS, TH_WORDS, TH_WORDS, TH_WORDS, TH_SIGNS, TH_NUMBERS, TH_LATIN, TH_PUNCT],
}


def rng_line(rng: random.Random, lang: str, words: int = 30) -> str:
    """A line of up to ``words`` pieces of ``lang``, drawn with ``rng``."""
    pools, seps = RNG_POOLS[lang], SEPS[lang]
    out = [rng.choice(rng.choice(pools))]
    for _ in range(rng.randint(0, words)):
        piece = rng.choice(rng.choice(pools))
        out.append(rng.choice(seps) + (piece.title() if rng.random() < 0.1 else piece))
    return "".join(out) + rng.choice(ENDINGS[lang])


def thai_run(rng: random.Random, size: int, p: float) -> str:
    """PyThaiNLP words without spaces (newmm-safe cuts a run of 140 characters or more in its
    100-140 window, at the last space there, else at its longest token), and now and then (with
    probability ``p``) a space, a number, Latin letters or punctuation."""
    out, n = [], 0
    other = [" ", " ", " ", "\n", *TH_NUMBERS, *TH_LATIN, *TH_PUNCT, *TH_SIGNS]
    while n < size:
        w = rng.choice(other) if rng.random() < p else rng.choice(TH_WORDS)
        out.append(w)
        n += len(w)
    return "".join(out)


@st.composite
def long_documents(draw: st.DrawFn, lang: str) -> str:
    """Several kB: random lines (some repeated), one long line, drawn lines repeated; for Thai
    also unspaced runs past newmm's safe-mode window."""
    rng = random.Random(draw(st.integers(0, 2**32 - 1), label="seed"))
    shapes = ["lines", "one-line", "repeated"] + (["thai-run", "thai-run"] if lang == TH else [])
    shape = draw(st.sampled_from(shapes), label="shape")
    size = draw(st.integers(141 if shape == "thai-run" else 2_000, 12_000), label="chars")
    if shape == "thai-run":
        p = draw(st.sampled_from([0.0, 0.0, 0.005, 0.03, 0.15]), label="p")
        text = thai_run(rng, size, p)
    elif shape == "repeated":
        base = draw(st.lists(lines(lang), min_size=1, max_size=6))
        unit = "\n".join(base) + "\n"
        if not unit.strip():
            unit += "Hà Nội là thủ đô.\n" if lang == VI else "กรุงเทพมหานครเป็นเมืองหลวง\n"
        n = size // len(unit) + 1
        text = "".join(unit if i % 7 else f"{unit}{i}. " for i in range(n))
    else:
        sep = " " if shape == "one-line" else None
        parts: list[str] = []
        while sum(map(len, parts)) < size:
            line = rng.choice(parts) if parts and rng.random() < 0.15 else rng_line(rng, lang)
            parts.append(line + (sep or rng.choice(LINE_SEPS)))
        text = "".join(parts)
    _event(f"{lang} {shape}: {len(text) // 1000} k chars")
    return text


# ---- the language table ------------------------------------------------------------------


def test_vietnamese_codes() -> None:
    langs = pf.languages()
    codes = [c for c, v in langs.items() if (v.tokenizer, v.arg) == ("SpaCyTokenizer", "vi")]
    assert codes == VI_CODES
    for code in VI_CODES:
        assert langs[code] == pf.Language("SpaCyTokenizer", "vi", True), code
        assert dd.languages()[code].native


def test_thai_codes() -> None:
    langs = pf.languages()
    assert [c for c, v in langs.items() if v.tokenizer == "ThaiTokenizer"] == TH_CODES
    for code in TH_CODES:
        assert langs[code] == pf.Language("ThaiTokenizer", "", THAI), code
        assert dd.languages()[code].native is THAI


def test_native_counts() -> None:
    # 4,025 without data files; Japanese's four codes with a SudachiDict, Thai's twenty with
    # PyThaiNLP.
    langs = pf.languages()
    assert sum(v.native for v in langs.values()) == 4025 + 4 * SUDACHI + 20 * THAI


VI_DOCS = [
    "Hà Nội là thủ đô của Việt Nam. Thành phố có khoảng 8,4 triệu dân!\nHôm nay trời đẹp.",
    "",
    "ngắn",
    "Tôi ăn phở. " * 20,
    "\n".join(["Dòng này đủ dài để được đếm, với nhiều chữ trong đó."] * 6),
    "Con mèo ngủ. Trời mưa. Rất lạnh. Hết.\n" * 3 + "Một thứ khác.",
    "TP. Hồ Chí Minh, Mr. Nam và ThS. Lê đến lúc 08:30 ==> giá 1.000.000,50 đồng... "
    "https://vnexpress.net/a?b=1 info@vnexpress.net",
    "Vie\u0323\u0302t Nam (NFD) và Việt Nam (NFC); ĐẠI HỌC QUỐC GIA HÀ NỘI 😀 LENGTH",
]
TH_DOCS = [
    "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย มีประชากรประมาณ 10,539,000 คน!\nวันนี้อากาศดีมาก",
    "",
    "สั้น",
    "ฉันกินข้าว. " * 20,
    "\n".join(["บรรทัดนี้ยาวพอที่จะนับได้ และมีคำมากมายอยู่ในนั้น."] * 6),
    "แมวนอน. ฝนตก. หนาวมาก. จบ.\n" * 3 + "อีกเรื่องหนึ่ง",
    "ราคา 1,000.50 บาท เวลา 12:30น ที่ www.sanook.com หรือ info@example.co.th ๆ ฯลฯ COVID-19",
    "ภาษาไทย" * 40 + " และ " + "สวนสาธารณะ" * 30,  # past newmm's safe-mode window
]
DOCS = {VI: VI_DOCS, TH: TH_DOCS}
ALL_FUNCS: list[Callable[..., Any]] = FILTERS.ALL_FUNCS


def chain(lang: str, code: str | None = None) -> list[Any]:
    """FineWeb-2's filters for ``lang``, and C4 (which rewrites the text) in front."""
    code = code or lang
    return [
        pf.C4Quality(language=code, min_num_sentences=1, filter_no_terminal_punct=False),
        *(step._replace(language=code) for step in FILTERS.fineweb2_steps(lang)),
    ]


@pytest.mark.parametrize("lang", NATIVE_LANGS)
def test_codes_share_the_tokenizer(lang: str) -> None:
    docs = DOCS[lang]
    want = pf.apply(docs, chain(lang))
    sigs = dd.minhash_signatures(docs, language=lang, n_grams=2)
    for code in VI_CODES if lang == VI else TH_CODES:
        assert pf.apply(docs, chain(lang, code)) == want, code
        for f in ALL_FUNCS:
            assert f(docs, language=code) == f(docs, language=lang), (f, code)
        assert dd.minhash_signatures(docs, language=code, n_grams=2) == sigs, code


def test_thai_native_with_pythainlp_else_opt_in() -> None:
    if THAI:
        assert pf.gopher_quality_reasons(TH_DOCS[0], language=TH) == "gopher_short_doc"
        return
    for f in ALL_FUNCS:
        with pytest.raises(NotImplementedError) as e:
            f(["ภาษาไทย"], language=TH)
        for name in [
            "ThaiTokenizer()",
            "PyThaiNLP",
            "python_tokenizer=True",
            "Or install pythainlp (5.4.0), and pyroid tokenizes it natively.",
        ]:
            assert name in str(e.value), (f, name)
    with pytest.raises(NotImplementedError, match="python_tokenizer=True"):
        pf.apply([], chain(TH))
    with pytest.raises(NotImplementedError, match="MinHash still run in Rust"):
        dd.minhash_signatures(["ภาษาไทย"], language=TH)
    with pytest.raises(NotImplementedError, match="language 'tha_Thai'") as e:
        dd.SentenceDedup(language=TH)
    assert "; or install pythainlp (5.4.0), and pyroid tokenizes it natively" in str(e.value)
    # Lines need no tokenizer.
    d = dd.SentenceDedup(language=TH, split_sentences=False, **SENT.NO_MIN)
    assert d.fit(["a\nb\nc", "a\nb\nc"]) == (["a\nb\nc", ""], [True, False])


# ---- live datatrove ------------------------------------------------------------------

#: ("ok", keep, reason, text) or ("raise", exception type name).
Outcome = tuple[Any, ...]


class Reference:
    """datatrove's tokenizer and filters for one language (test_filters_multilingual's harness)."""

    def __init__(self, lang: str) -> None:
        self.lang = lang
        self.dt = FILTERS.Datatrove()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)  # pyvi 0.1.1: invalid escapes
            self.tok: Any = self.dt.wt.load_word_tokenizer(lang)

    def run(self, steps: Sequence[Any], text: str) -> Outcome:
        out: Outcome = self.dt.run(steps, text)
        return out


def _skip_without(lang: str) -> None:
    """Skip unless pyroid tokenizes ``lang`` natively and datatrove can here."""
    if lang == TH and not THAI:
        pytest.skip("no PyThaiNLP: Thai is not native (the opt-in path is tested above)")
    for module in MODULES[lang]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            pytest.importorskip(module)


@pytest.fixture(scope="module", params=LANGS)
def ref(request: pytest.FixtureRequest) -> Reference:
    lang: str = request.param
    _skip_without(lang)
    return Reference(lang)


def _short(o: Outcome) -> Outcome:
    if o[0] == "ok" and isinstance(o[3], str) and len(o[3]) > 120:
        return (*o[:3], f"<{len(o[3])} chars> {o[3][:60]!r}...{o[3][-40:]!r}")
    return o


def _first_difference(a: str, b: str) -> str:
    i = next((k for k, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
    lo = max(i - 30, 0)
    return f"at {i} of {len(a)}/{len(b)}: {a[lo : i + 30]!r} vs {b[lo : i + 30]!r}"


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
                f"{steps}\ndocument {i} ({len(docs[i])} chars): {docs[i][:400]!r}\n"
                f"pyroid {_short(g)} != datatrove {_short(w)}{where}"
            )


#: A C4 that keeps every sentence: its output shows where datatrove's sentence spans end.
C4_KEEP_ALL: dict[str, Any] = {
    "filter_no_terminal_punct": False,
    "min_num_sentences": -1,
    "min_words_per_line": -1,
    "max_word_length": -1,
    "filter_lorem_ipsum": False,
    "filter_javascript": False,
    "filter_curly_bracket": False,
    "filter_policy": False,
}


def configs(kind: str, lang: str) -> list[Any]:
    """test_filters_multilingual's spread, FineWeb-2's settings for ``lang``, and C4s that keep
    every sentence (split_paragraph=False: datatrove's sentences) or every line."""
    out: list[Any] = FILTERS.configs(kind, lang, STOPS[lang][:6])
    fw2 = dict(zip(["gr", "fw", "gq"], FILTERS.fineweb2_steps(lang), strict=True))
    if kind in fw2:
        out.append(fw2[kind])
    if kind == "c4":
        out.append(pf.C4Quality(language=lang, split_paragraph=False, **C4_KEEP_ALL))
        out.append(pf.C4Quality(language=lang, **C4_KEEP_ALL))
        out.append(pf.C4Quality(language=lang, split_paragraph=False, min_words_per_line=2))
        out.append(pf.C4Quality(language=lang, max_word_length=6, filter_no_terminal_punct=False))
    return out


KINDS = ["gq", "gr", "fw", "c4"]


@pytest.mark.parametrize("kind", KINDS)
@examples(40)
@given(data=st.data())
def test_filters_match_datatrove(ref: Reference, kind: str, data: st.DataObject) -> None:
    cfg = data.draw(st.sampled_from(configs(kind, ref.lang)), label="config")
    docs = data.draw(st.lists(documents(ref.lang), min_size=1, max_size=4), label="docs")
    check(ref, [cfg], docs)


@examples(40)
@given(data=st.data())
def test_chains_match_datatrove(ref: Reference, data: st.DataObject) -> None:
    lang = ref.lang
    steps = data.draw(
        st.sampled_from(
            [chain(lang), FILTERS.fineweb2_steps(lang)]
            + [c(lang, STOPS[lang]) for c in FILTERS.CHAINS]
        )
    )
    docs = data.draw(st.lists(documents(lang), min_size=1, max_size=4), label="docs")
    check(ref, steps, docs)


@pytest.mark.parametrize("kind", KINDS)
@examples(6)
@given(data=st.data())
def test_long_documents_match_datatrove(ref: Reference, kind: str, data: st.DataObject) -> None:
    cfg = data.draw(st.sampled_from(configs(kind, ref.lang)), label="config")
    doc = data.draw(long_documents(ref.lang), label="doc")
    check(ref, [cfg], [doc])


def _c4_lines(text: str, cfg: Any) -> Outcome:
    try:
        return ("ok", pf.c4_lines(text, cfg))
    except Exception as e:
        return ("raise", type(e).__name__)


@examples(30)
@given(data=st.data())
def test_c4_lines_match_datatrove(ref: Reference, data: st.DataObject) -> None:
    """``c4_lines``: datatrove's ``doc.text`` after C4QualityFilter, kept or not."""
    cfg = data.draw(st.sampled_from(configs("c4", ref.lang)), label="config")
    docs = data.draw(st.lists(documents(ref.lang), min_size=1, max_size=4), label="docs")
    want = [ref.run([cfg], d) for d in docs]
    want_lines = [("ok", w[3]) if w[0] == "ok" else w for w in want]
    assert [_c4_lines(d, cfg) for d in docs] == want_lines
    if all(w[0] == "ok" for w in want):
        assert pf.c4_lines(docs, cfg) == [w[3] for w in want]


OFF = dd.TextNormConfig(
    lowercase=False,
    norm_whitespace=False,
    remove_punctuation=False,
    norm_unicode_diacritics=False,
    norm_numbers=False,
)


def same_words(ref: Reference, text: str) -> None:
    """pyroid's words for ``text`` are datatrove's: MinHash shingles of them, without
    normalization (``simplify_text`` then only strips).

    One shingle of all the words checks how many there are and their text joined with spaces.
    Vietnamese words have spaces in them, so the 1-gram and 2-gram signatures over 256 hash
    functions check where the words end (a boundary elsewhere changes the shingle set).
    """
    words = ref.tok.word_tokenize(text.strip())
    for n, per in ((max(len(words), 1), 1), (1, 256), (2, 256)):
        cfg = dd.MinhashConfig(n_grams=n, norm_config=OFF, num_buckets=1, hashes_per_bucket=per)
        got = dd.minhash_signatures(text, cfg, language=ref.lang)
        want = dd.minhash_signatures(text, cfg, language=ref.tok)
        assert got == want, (n, len(words), words[:40], text[:300])


@examples(100)
@given(data=st.data())
def test_words_match_datatrove(ref: Reference, data: st.DataObject) -> None:
    same_words(ref, data.draw(documents(ref.lang), label="doc"))


@examples(8)
@given(data=st.data())
def test_words_of_long_documents_match_datatrove(ref: Reference, data: st.DataObject) -> None:
    same_words(ref, data.draw(long_documents(ref.lang), label="doc"))


def sentence_hashes(lang: str, texts: list[str], n: int, norm: Any) -> None:
    """Stage 1's (hash, sentence) pairs, from ``sent_tokenize``, are datatrove's."""
    kw: dict[str, Any] = {"n_sentences": n, "split_sentences": True}
    step = SENT.dt_sd.SentenceDedupSignature(
        output_folder="/nonexistent",
        config=SENT.dt_config(("xxhash", 64), norm, **kw),
        language=lang,
    )
    native = dd._native_sent(SENT.py_config(("xxhash", 64), norm, **kw), lang)
    want = [
        [(h, s) for h, _, s in step.get_hashes(SENT.Document(text=t, id="x"), 0)] for t in texts
    ]
    got = [[tuple(x) for x in doc] for doc in dd._native._sentence_hashes(texts, native)]
    assert got == want


@pytest.fixture(scope="module")
def sd(ref: Reference) -> Any:
    """The sentence-dedup harness (skips without it)."""
    if not SENT.HAVE_DT:
        pytest.skip("needs datatrove 0.10.1 and spaCy")
    return SENT


@examples(60)
@given(data=st.data())
def test_sentences_match_datatrove(ref: Reference, sd: Any, data: st.DataObject) -> None:
    texts = data.draw(st.lists(documents(ref.lang), min_size=1, max_size=4), label="texts")
    n = data.draw(st.integers(1, 3), label="n")
    norm = data.draw(st.one_of(st.just(OFF), SENT.norm_configs), label="norm")
    sentence_hashes(ref.lang, texts, n, norm)


@examples(6)
@given(data=st.data())
def test_sentences_of_long_documents_match_datatrove(
    ref: Reference, sd: Any, data: st.DataObject
) -> None:
    doc = data.draw(long_documents(ref.lang), label="doc")
    sentence_hashes(ref.lang, [doc], data.draw(st.integers(1, 3), label="n"), OFF)


VI_SENTENCES = [
    "Con mèo đang ngủ trên tấm thảm.", "Trời lại mưa rồi!", "Bạn có khỏe không?",
    "Giá tăng 3,5% trong tháng Năm.", "Ông Nam đi Hà Nội hôm qua.", "TP. Hồ Chí Minh rất đông.",
    "Anh ấy nói: “Xin chào”.", "Đây là quyển sách", "Cảm ơn nhiều!!", "Thật không?",
    "Mr. Nam và ThS. Lê họp lúc 08:30...",
]  # fmt: skip
TH_SENTENCES = [
    "แมวนอนอยู่บนเสื่อ", "วันนี้ฝนตกอีกแล้ว!", "คุณสบายดีไหม?", "ราคาเพิ่มขึ้น 3.5% ในเดือนพฤษภาคม.",
    "ฉันไปกรุงเทพมหานครเมื่อวานนี้ ", "อากาศดีมาก ", "เขาพูดว่า “สวัสดี”", "นี่คือหนังสือ",
    "ขอบคุณมากครับ!!", "จริงหรือ?", "เวลา 12:30น ที่ www.sanook.com.",
]  # fmt: skip
SENTENCES = {VI: VI_SENTENCES, TH: TH_SENTENCES}
SENT_SEPS = ["", "", " ", "  ", "\n", "\n\n", "　", " \n", "\r\n"]


@st.composite
def corpora(draw: st.DrawFn, lang: str) -> list[str]:
    """Small corpora whose documents share sentences."""
    pool = draw(st.lists(st.sampled_from(SENTENCES[lang]) | lines(lang), min_size=1, max_size=8))
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


def stages(sd: Any, lang: str, texts: list[str], kw: dict[str, Any], work: Path) -> None:
    """datatrove's three stages, pyroid's drop-ins (natively) and SentenceDedup agree."""
    import pyroid.integrations.datatrove as pdt

    cfg = sd.dt_config(**kw)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pyroid.FallbackWarning)
        want = sd.run_stages(sd.dt_sd, texts, cfg, work / "dt", language=lang)
        got = sd.run_stages(pdt, texts, cfg, work / "pr", language=lang, batch_size=2)
        text, keep = dd.SentenceDedup(sd.py_config(**kw), language=lang).fit(texts)
    for key in ("sig", "dup", "excl"):
        assert got[key] == want[key], key
    for (gi, gt), (wi, wt) in zip(got["kept"], want["kept"], strict=True):
        assert (gi, len(gt)) == (wi, len(wt)), _first_difference(gt, wt)
        assert gt == wt, _first_difference(gt, wt)
    assert len(got["kept"]) == len(want["kept"])
    kept = dict(want["kept"])
    assert keep == [str(i) in kept for i in range(len(texts))]
    assert [t for t, k in zip(text, keep, strict=True) if k] == [t for _, t in want["kept"]]


@examples(30)
@given(data=st.data())
def test_sentence_dedup_matches_datatrove(
    ref: Reference, sd: Any, data: st.DataObject, tmp_path_factory: pytest.TempPathFactory
) -> None:
    texts = data.draw(corpora(ref.lang), label="texts")
    n = data.draw(st.integers(1, 3), label="n")
    mins = data.draw(st.sampled_from(SD_MINS), label="mins")
    stages(sd, ref.lang, texts, {"n_sentences": n, **mins}, tmp_path_factory.mktemp("vt"))


@examples(4)
@given(data=st.data())
def test_sentence_dedup_of_long_documents_matches_datatrove(
    ref: Reference, sd: Any, data: st.DataObject, tmp_path_factory: pytest.TempPathFactory
) -> None:
    doc = data.draw(long_documents(ref.lang), label="doc")
    texts = data.draw(st.sampled_from([[doc], [doc[:3000], doc], [doc, doc[-5000:]]]))
    mins = data.draw(st.sampled_from(SD_MINS))
    n = data.draw(st.integers(1, 3))
    stages(sd, ref.lang, texts, {"n_sentences": n, **mins}, tmp_path_factory.mktemp("vl"))


@st.composite
def families(draw: st.DrawFn, lang: str) -> list[str]:
    """Near-duplicates: inserted, deleted, respaced and re-punctuated pieces, case changes."""
    docs: list[str] = []
    for _ in range(draw(st.integers(1, 3))):
        base = draw(documents(lang))
        docs.append(base)
        for _ in range(draw(st.integers(0, 3))):
            i = draw(st.integers(0, len(base)))
            op = draw(st.sampled_from(["insert", "delete", "space", "punct", "upper"]))
            if op == "insert":
                docs.append(base[:i] + draw(PIECES[lang]()) + base[i:])
            elif op == "delete":
                docs.append(base[:i] + base[min(i + draw(st.integers(1, 8)), len(base)) :])
            elif op == "space":
                docs.append(base[:i] + draw(st.sampled_from(WS[lang])) + base[i:])
            elif op == "punct":
                docs.append(base[:i] + draw(st.sampled_from(VI_PUNCT + TH_PUNCT)) + base[i:])
            else:
                docs.append(base.upper())
    return draw(st.permutations(docs))


@pytest.fixture(scope="module")
def dt_minhash(ref: Reference, tmp_path_factory: pytest.TempPathFactory) -> Any:
    return DEDUP.Datatrove(tmp_path_factory.mktemp("datatrove-minhash"))


@examples(40)
@given(data=st.data())
def test_minhash_matches_datatrove(ref: Reference, dt_minhash: Any, data: st.DataObject) -> None:
    texts = data.draw(families(ref.lang), label="texts")
    hc = data.draw(st.sampled_from(DEDUP.HASHES), label="hash")
    n = data.draw(st.sampled_from([1, 2, 5]), label="n_grams")
    DEDUP.check(dt_minhash, ref.lang, texts, hc, n, python=False)


@examples(6)
@given(data=st.data())
def test_minhash_of_long_documents_matches_datatrove(
    ref: Reference, dt_minhash: Any, data: st.DataObject
) -> None:
    doc = data.draw(long_documents(ref.lang), label="doc")
    n = data.draw(st.sampled_from([1, 3, 5]), label="n_grams")
    kw: dict[str, Any] = {"n_grams": n, "num_buckets": 6, "hashes_per_bucket": 2}
    texts = [doc, doc[: len(doc) // 2] + "." + doc[len(doc) // 2 :]]
    want = dt_minhash.signatures(dt_minhash.config("xxhash", 64, **kw), ref.lang, texts)
    assert dd.minhash_signatures(texts, dd.MinhashConfig(**kw), language=ref.lang) == want


def test_python_tokenizer_is_identical(ref: Reference) -> None:
    rng = random.Random(5)
    lang = ref.lang
    docs = [*DOCS[lang], *(rng_line(rng, lang) for _ in range(20))]
    docs.append("\n".join(rng_line(rng, lang) for _ in range(300)))
    for cfg in [*(configs(k, lang)[0] for k in KINDS), *chain(lang)]:
        assert pf.apply(docs, [cfg], python_tokenizer=True) == pf.apply(docs, [cfg]), cfg
    for n in (1, 5):
        native = dd.minhash_signatures(docs, language=lang, n_grams=n)
        assert (
            dd.minhash_signatures(docs, language=lang, n_grams=n, python_tokenizer=True) == native
        )
        assert dd.minhash_signatures(docs, language=ref.tok, n_grams=n) == native


# ---- the FineWeb-2 slices --------------------------------------------------------------

DATA = Path(os.environ.get("PYROID_DATA", "/data"))


def fineweb2_texts(lang: str) -> list[str]:
    """The pinned FineWeb-2 slice of ``lang`` (250 documents; bench/manifest.json)."""
    pq: Any = pytest.importorskip("pyarrow.parquet")
    path = DATA / "fineweb2" / f"{lang}.parquet"
    if not path.is_file():
        pytest.skip(f"FineWeb-2 slice not found: {path}")
    texts: list[str] = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    return texts


def test_fineweb2_filters_match_datatrove(ref: Reference) -> None:
    """FineWeb-2's pipeline on its own documents, then each filter and the configs spread."""
    texts = fineweb2_texts(ref.lang)
    steps = FILTERS.fineweb2_steps(ref.lang)
    check(ref, steps, texts)
    for kind in KINDS:
        for cfg in configs(kind, ref.lang):
            check(ref, [cfg], texts)
    # FineWeb-2 kept these documents; most still pass its filters.
    assert sum(pf.apply(texts, steps).keep) > len(texts) // 2


def test_fineweb2_words_and_sentences_match_datatrove(ref: Reference, sd: Any) -> None:
    texts = fineweb2_texts(ref.lang)
    for t in texts:
        same_words(ref, t)
    for n in (1, 3):
        sentence_hashes(ref.lang, texts, n, OFF)
        sentence_hashes(ref.lang, texts, n, dd.TextNormConfig())


def test_fineweb2_minhash_matches_datatrove(ref: Reference, dt_minhash: Any) -> None:
    """FineWeb-2's MinhashConfig (xxhash/64, 14 buckets of 8, 5-grams), with a near-duplicate
    of every fifth document: the signatures and datatrove's removals."""
    texts = fineweb2_texts(ref.lang)
    texts = texts + [t[: len(t) // 2] + t[len(t) // 2 + 1 :] for t in texts[::5]]
    want_cfg = dt_minhash.config("xxhash", 64, num_buckets=14, hashes_per_bucket=8, n_grams=5)
    cfg = dd.MinhashConfig(num_buckets=14, hashes_per_bucket=8, n_grams=5)
    want = dt_minhash.signatures(want_cfg, ref.lang, texts)
    got = dd.minhash_signatures(texts, cfg, language=ref.lang)
    bad = [i for i, (g, w) in enumerate(zip(got, want, strict=True)) if g != w]
    assert not bad, f"{len(bad)}/{len(texts)} signatures differ: {bad[:5]}"
    removed = dt_minhash.removed(want_cfg, ref.lang, texts)["removed"]
    assert len(removed) >= 25  # the near-duplicates are found
    keep = dd.minhash_dedup(texts, cfg, language=ref.lang)
    assert keep == [i not in removed for i in range(len(texts))]


def test_fineweb2_sentence_dedup_matches_datatrove(ref: Reference, sd: Any, tmp_path: Path) -> None:
    texts = fineweb2_texts(ref.lang)[:80]
    # Copies of halves, and documents sharing a paragraph, so spans are cut out.
    texts += [t[: len(t) // 2] for t in texts[:10]] + [texts[0] + "\n" + t for t in texts[10:20]]
    for i, mins in enumerate(SD_MINS):
        stages(sd, ref.lang, texts, {"n_sentences": 3, **mins}, tmp_path / str(i))


# ---- containers, threads, errors (no datatrove needed) -------------------------------------


def sample_docs(lang: str, n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        ls = [rng_line(rng, lang) for _ in range(rng.randint(0, 10))]
        out.append(rng.choice(["\n", "\n\n", " "]).join(ls))
    return out


@pytest.mark.parametrize("lang", NATIVE_LANGS)
def test_container_kinds(lang: str) -> None:
    docs = DOCS[lang]
    steps = chain(lang)
    res = pf.apply(docs, steps)
    assert isinstance(res.keep, list)
    assert pf.apply(tuple(docs), steps) == tuple(map(tuple, res))
    for i, d in enumerate(docs):
        one = pf.apply(d, steps)
        assert one == (res.keep[i], res.reason[i], res.text[i])
        if one.text == d:
            assert one.text is d  # unchanged texts are the original objects
    for f in ALL_FUNCS:
        many = f(docs, language=lang)
        assert f(tuple(docs), language=lang) == tuple(many)
        assert [f(d, language=lang) for d in docs] == many
    sigs = dd.minhash_signatures(docs, language=lang)
    assert dd.minhash_signatures(tuple(docs), language=lang) == sigs
    assert [dd.minhash_signatures(d, language=lang) for d in docs] == sigs
    assert sigs[1] is None  # no shingles
    keep = dd.minhash_dedup([*docs, docs[0]], language=lang)
    assert keep == [True] * len(docs) + [False]
    assert dd.minhash_dedup(tuple(docs), language=lang) == (True,) * len(docs)
    res_sd = dd.sentence_dedup([*docs, docs[5]], language=lang, **SENT.NO_MIN)
    assert res_sd.keep[-1] is False
    assert res_sd.text[3] != docs[3]  # its own repeats too
    tup = dd.sentence_dedup((*docs, docs[5]), language=lang, **SENT.NO_MIN)
    assert tup == (tuple(res_sd.text), tuple(res_sd.keep))
    assert tup.text[0] is docs[0]
    assert dd.sentence_dedup(docs[0], language=lang, **SENT.NO_MIN) == (docs[0], True)


@pytest.mark.parametrize("lang", NATIVE_LANGS)
@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("chunks", [1, 3])
def test_arrow_without_pyarrow(lang: str, large: bool, chunks: int) -> None:
    docs = DOCS[lang]
    values: list[str | None] = [*docs[:3], None, *docs[3:], None]
    col = _testing.column(values, large=large, chunks=chunks, name="text")
    plain = [v for v in values if v is not None]

    def spread(xs: Sequence[Any]) -> list[Any]:
        it = iter(xs)
        return [None if v is None else next(it) for v in values]

    want = pf.apply(plain, chain(lang))
    res = pf.apply(col, chain(lang))
    assert res.keep.to_pylist() == spread(want.keep)
    assert res.reason.to_pylist() == spread(want.reason)
    assert res.text.to_pylist() == spread(want.text)
    assert res.text.type == ("large_string" if large else "string")
    assert res.keep.name == "text"
    for f in ALL_FUNCS:
        assert f(col, language=lang).to_pylist() == spread(f(plain, language=lang))
    assert dd.minhash_signatures(col, language=lang) == spread(
        dd.minhash_signatures(plain, language=lang)
    )
    mask = dd.minhash_dedup(col, language=lang)
    assert mask.to_pylist() == spread(dd.minhash_dedup(plain, language=lang))
    text, keep = dd.sentence_dedup(col, language=lang, **SENT.NO_MIN)
    want_sd = dd.sentence_dedup(plain, language=lang, **SENT.NO_MIN)
    assert text.to_pylist() == spread(want_sd.text)
    assert keep.to_pylist() == spread(want_sd.keep)


@pytest.mark.parametrize("lang", NATIVE_LANGS)
def test_pyarrow(pa: Any, lang: str) -> None:
    docs = DOCS[lang]
    arr = pa.chunked_array([docs[:4], [None, *docs[4:]]], type=pa.string_view())
    want = pf.apply(docs, chain(lang))
    res = pf.apply(arr, chain(lang))
    assert res.keep.to_pylist() == [*want.keep[:4], None, *want.keep[4:]]
    assert pa.chunked_array(res.text).to_pylist() == [*want.text[:4], None, *want.text[4:]]
    sliced = pa.array(docs, type=pa.large_string()).slice(2)
    got = pf.gopher_repetition_reasons(sliced, language=lang)
    assert got.to_pylist() == pf.gopher_repetition_reasons(docs[2:], language=lang)
    assert dd.minhash_signatures(sliced, language=lang) == dd.minhash_signatures(
        docs[2:], language=lang
    )


@pytest.mark.parametrize("lang", NATIVE_LANGS)
@pytest.mark.parametrize("threads", [None, 1, 8])
def test_threads(lang: str, threads: int | None) -> None:
    docs = sample_docs(lang, 300, seed=1)
    steps = chain(lang)
    assert pf.apply(docs, steps, threads=threads) == pf.apply(docs, steps, threads=1)
    for f in (pf.gopher_quality_reasons, pf.c4_lines):
        assert f(docs, language=lang, threads=threads) == f(docs, language=lang, threads=1)
    want = dd.minhash_signatures(docs, language=lang, threads=1)
    assert dd.minhash_signatures(docs, language=lang, threads=threads) == want
    d = dd.MinHashDedup(language=lang, threads=threads)
    d.add(docs[:150])
    d.add(docs[150:] + docs[:20])
    assert d.keep_mask() == dd.minhash_dedup(docs + docs[:20], language=lang, threads=1)
    got = dd.SentenceDedup(language=lang, threads=threads, **SENT.NO_MIN).fit(docs)
    assert got == dd.SentenceDedup(language=lang, threads=1, **SENT.NO_MIN).fit(docs)


@pytest.mark.parametrize("lang", NATIVE_LANGS)
@pytest.mark.timeout(300)
def test_2000_documents(lang: str) -> None:
    docs = sample_docs(lang, 2000, seed=2)
    res = pf.apply(docs, chain(lang))
    assert len(res.keep) == 2000
    assert len(set(res.reason)) > 3
    sigs = dd.minhash_signatures(docs, language=lang)
    assert len(sigs) == 2000
    # 100 copies at the end: removed (those with shingles); the others as without them.
    keep = dd.minhash_dedup(docs + docs[:100], language=lang)
    assert keep[:2000] == dd.minhash_dedup(docs, language=lang)
    assert not any(k for k, s in zip(keep[2000:], sigs, strict=False) if s is not None)
    text, kept = dd.sentence_dedup(docs + docs[:100], language=lang)
    assert (text[:2000], kept[:2000]) == dd.sentence_dedup(docs, language=lang)
    assert sum(t != d for t, d in zip(text[2000:], docs, strict=False)) > 50


@pytest.mark.parametrize("lang", NATIVE_LANGS)
def test_input_errors(lang: str) -> None:
    word = DOCS[lang][2]
    with pytest.raises(pyroid.InputError, match="index 1"):
        pf.gopher_quality([word, 3], language=lang)  # type: ignore[list-item]
    with pytest.raises(pyroid.InputError):
        pf.apply(b"bytes", chain(lang))  # type: ignore[arg-type]
    with pytest.raises(pyroid.InputError):
        dd.minhash_signatures([word, None], language=lang)  # type: ignore[list-item]
    with pytest.raises(pyroid.InputError):
        dd.SentenceDedup(language=lang).add([1])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="threads"):
        pf.apply(DOCS[lang] * 10, chain(lang), threads=0)


@pytest.mark.parametrize("lang", NATIVE_LANGS)
@examples(15)
@given(data=st.data(), pos=st.integers(0, 10_000), cs=st.characters(categories=["Cs"]))
def test_lone_surrogates_raise_unicode_encode_error(
    lang: str, data: st.DataObject, pos: int, cs: str
) -> None:
    doc = data.draw(documents(lang), label="doc")
    at = pos % (len(doc) + 1)
    bad = doc[:at] + cs + doc[at:]
    word = DOCS[lang][2]
    for f in ALL_FUNCS:
        with pytest.raises(UnicodeEncodeError) as e:
            f([word, bad], language=lang)
        assert "index 1" in _notes(e.value), _notes(e.value)
    with pytest.raises(UnicodeEncodeError):
        pf.apply(bad, chain(lang))
    with pytest.raises(UnicodeEncodeError):
        dd.minhash_signatures([word, bad], language=lang)
    with pytest.raises(UnicodeEncodeError):
        dd.MinHashDedup(language=lang).add([bad])
    with pytest.raises(UnicodeEncodeError):
        dd.SentenceDedup(language=lang).add([bad])


# ---- PyThaiNLP, in child interpreters --------------------------------------------------

THAI_CALLS = """
import os, sys
import pyroid, pyroid.filters as pf, pyroid.dedup as dd
calls = [
    ("tha_Thai", lambda: pf.gopher_quality(["ภาษาไทย"], language="tha_Thai")),
    ("th", lambda: pf.apply(["ภาษาไทย"], [pf.C4Quality(language="th")])),
    ("tha", lambda: dd.minhash_signatures(["ภาษาไทย"], language="tha")),
    ("th_Thai", lambda: dd.MinHashDedup(language="th_Thai").fit(["ภาษาไทย"])),
    ("tha_Thai", lambda: dd.SentenceDedup(language="tha_Thai").fit(["ภาษาไทย."])),
]
"""

NO_PYTHAINLP = (
    """
import sys
sys.modules["pythainlp"] = None  # not installed
"""
    + THAI_CALLS
    + """
import importlib.util
langs = pf.languages()
assert not any(v.native for v in langs.values() if v.tokenizer == "ThaiTokenizer")
assert langs["vie_Latn"].native  # pyvi's data ships with pyroid
ja = "PYROID_SUDACHI_DICT" in os.environ or importlib.util.find_spec("sudachidict_core") is not None
assert sum(v.native for v in langs.values()) == 4025 + 4 * ja, sum(v.native for v in langs.values())
for code, call in calls:
    try:
        call()
    except NotImplementedError as e:
        print("NotImplementedError:", code, str(e).replace("\\n", " "))
    else:
        raise SystemExit(f"{code}: no NotImplementedError")
d = dd.SentenceDedup(
    language="tha_Thai", split_sentences=False, min_doc_words=0, min_num_sentences=0
)
assert d.fit(["a\\nb\\nc", "a\\nb\\nc"]) == (["a\\nb\\nc", ""], [True, False])
assert pf.gopher_quality_reasons("Hà Nội là thủ đô.", language="vie_Latn") == "gopher_short_doc"
if all(importlib.util.find_spec(m) for m in ("datatrove", "spacy")):
    try:
        pf.gopher_quality(["ภาษาไทย"], language="tha_Thai", python_tokenizer=True)
    except pyroid.PyroidError as e:  # datatrove does not find PyThaiNLP either
        print("PyroidError:", e)
    else:
        raise SystemExit("python_tokenizer=True found PyThaiNLP")
"""
)


@pytest.mark.thread_unsafe
def test_no_pythainlp(run_python: Callable[..., Any]) -> None:
    r = run_python(NO_PYTHAINLP, env={"PYROID_PYTHAINLP_CORPUS": None})
    assert r.returncode == 0, r.stderr
    out = r.stdout.splitlines()
    errors = [ln.split(" ", 2)[1:] for ln in out if ln.startswith("NotImplementedError:")]
    assert len(errors) == 5, out
    for code, msg in errors[:4]:  # the filters and MinHash
        for part in (
            f"'{code}'",
            "ThaiTokenizer() (PyThaiNLP)",
            "python_tokenizer=True",
            "Or install pythainlp (5.4.0), and pyroid tokenizes it natively.",
        ):
            assert part in msg, (part, msg)
    code, msg = errors[4]  # SentenceDedup: datatrove's steps, or lines
    for part in (f"'{code}'", "PyThaiNLP", "; or install pythainlp (5.4.0), and pyroid tokenizes"):
        assert part in msg, (part, msg)
    for ln in out:
        if ln.startswith("PyroidError:"):
            assert "pythainlp" in ln, ln
            assert "pip install" in ln, ln


def thai_corpus() -> Path:
    """The installed PyThaiNLP's corpus directory (skip without it)."""
    words = _package_file("pythainlp", "corpus", "words_th.txt")
    if words is None:
        pytest.skip("pythainlp is not installed")
    return words.parent


BAD_CORPUS = (
    THAI_CALLS
    + """
assert pf.languages()["tha_Thai"].native  # a directory to read: native until it fails to load
for path, want, part in CASES:
    os.environ["PYROID_PYTHAINLP_CORPUS"] = path
    for _, call in calls * 2:  # every use tries again
        try:
            call()
        except Exception as e:
            assert type(e).__name__ == want, (path, type(e), e)
            assert part in str(e), (path, part, e)
            msg = str(e)
        else:
            raise SystemExit(f"{path}: no {want}")
    print(want, msg)
# Other languages do not need the corpus.
assert pf.gopher_quality_reasons("Ein kurzer Satz.", language="deu_Latn") == "gopher_short_doc"
assert pf.gopher_quality_reasons("Hà Nội là thủ đô.", language="vie_Latn") == "gopher_short_doc"
"""
)


@pytest.mark.thread_unsafe
def test_missing_or_invalid_corpus(run_python: Callable[..., Any], tmp_path: Path) -> None:
    words, model = "words_th.txt", "sentenceseg_crfcut.model"
    only_words = tmp_path / "only-words"
    only_words.mkdir()
    (only_words / words).write_text("ภาษา\nไทย\n", encoding="utf-8")
    garbage = tmp_path / "garbage-model"
    garbage.mkdir()
    (garbage / words).write_text("ภาษา\nไทย\n", encoding="utf-8")
    (garbage / model).write_bytes(b"not a crfsuite model\n" * 100)
    empty = tmp_path / "empty-model"
    empty.mkdir()
    (empty / words).write_text("ภาษา\n", encoding="utf-8")
    (empty / model).write_bytes(b"")
    a_file = tmp_path / "a-file"
    a_file.write_text("x", encoding="utf-8")
    cases = [
        (str(tmp_path / "missing"), "OSError", "cannot read PyThaiNLP's"),
        (str(a_file), "OSError", "cannot read PyThaiNLP's"),
        (str(only_words), "OSError", model),
        (str(garbage), "ValueError", model),
        (str(empty), "ValueError", model),
    ]
    if _package_file("pythainlp", "corpus", model) is not None:
        real = thai_corpus()
        latin1 = tmp_path / "latin1-words"
        latin1.mkdir()
        (latin1 / words).write_bytes("ภาษา\n".encode("cp874"))  # not UTF-8
        shutil.copy(real / model, latin1 / model)
        cases.append((str(latin1), "ValueError", words))
        for size in (8, 100, 4096):  # truncated models
            cut = tmp_path / f"cut{size}"
            cut.mkdir()
            shutil.copy(real / words, cut / words)
            (cut / model).write_bytes((real / model).read_bytes()[:size])
            cases.append((str(cut), "ValueError", model))
    code = f"CASES = {cases!r}\n" + BAD_CORPUS
    r = run_python(code, env={"PYROID_PYTHAINLP_CORPUS": cases[0][0]})
    assert r.returncode == 0, r.stderr
    out = r.stdout.splitlines()
    assert len(out) == len(cases), out
    assert "cannot read PyThaiNLP's" in out[0], out[0]


ELSEWHERE = """
import json, os, sys
sys.modules["pythainlp"] = None  # only $$PYROID_PYTHAINLP_CORPUS
from pyroid.filters import C4Quality, FineWebQuality, GopherQuality, GopherRepetition
import pyroid.filters as pf, pyroid.dedup as dd
docs = $docs
steps = $steps
assert pf.languages()["tha_Thai"].native
res = pf.apply(docs, steps)
sigs = dd.minhash_signatures(docs, language="tha_Thai", n_grams=2)
sd = dd.SentenceDedup(language="tha_Thai", min_doc_words=0, min_num_sentences=0).fit(docs)
# Loaded once per process: the variable is not read again.
os.environ["PYROID_PYTHAINLP_CORPUS"] = "/nonexistent/corpus"
assert pf.apply(docs, steps) == res
assert dd.SentenceDedup(language="th").fit(docs).keep
print(json.dumps([list(res.keep), list(res.reason), list(res.text), sigs, list(sd.text)]))
"""


@pytest.mark.thread_unsafe
@needs_thai
def test_corpus_from_the_variable(run_python: Callable[..., Any]) -> None:
    """``$PYROID_PYTHAINLP_CORPUS`` with the package hidden: the results of the installed one."""
    docs = [*TH_DOCS, "ภาษาไทย" * 3000, *sample_docs(TH, 30, seed=6)]
    steps = chain(TH)
    code = string.Template(ELSEWHERE).substitute(
        docs=repr(docs), steps="[" + ", ".join(repr(s) for s in steps) + "]"
    )
    r = run_python(code, env={"PYROID_PYTHAINLP_CORPUS": str(thai_corpus())}, timeout=300)
    assert r.returncode == 0, r.stderr
    res = pf.apply(docs, steps)
    sigs = dd.minhash_signatures(docs, language=TH, n_grams=2)
    sd_text = dd.SentenceDedup(language=TH, **SENT.NO_MIN).fit(docs).text
    assert json.loads(r.stdout) == [list(res.keep), list(res.reason), list(res.text), sigs, sd_text]


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
            out.append(("f", pf.gopher_quality_reasons(docs, language="tha_Thai")))
        else:
            out.append(("m", dd.minhash_signatures(docs, language="th", n_grams=2)))
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
@needs_thai
def test_first_use_from_threads(run_python: Callable[..., Any]) -> None:
    docs = sample_docs(TH, 40, seed=3)
    code = string.Template(FROM_THREADS).substitute(docs=repr(docs))
    r = run_python(code, timeout=300)
    assert r.returncode == 0, r.stderr
    want = [
        pf.gopher_quality_reasons(docs, language=TH),
        dd.minhash_signatures(docs, language=TH, n_grams=2),
    ]
    assert json.loads(r.stdout) == want
