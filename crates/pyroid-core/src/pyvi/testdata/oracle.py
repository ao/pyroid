"""Record what spaCy's Vietnamese tokenizer (pyvi) and datatrove's word tokenizer return.

The goldens record the output of pyvi 0.1.1 (MIT), © 2016 Viet-Trung Tran, of spaCy 3.8.18
(MIT), © ExplosionAI GmbH, whose Vietnamese tokenizer is adapted from pyvi, and of datatrove
0.10.1 (Apache-2.0), © HuggingFace; pyvi's model runs in python-crfsuite 0.9.12 through
sklearn-crfsuite 0.5.0. The FineWeb-2 excerpts are from HuggingFaceFW/fineweb-2 (ODC-By 1.0),
subset vie_Latn.

pyroid's Vietnamese word tokenizer must give exactly what datatrove's
`load_word_tokenizer("vie_Latn")` gives: spaCy's `blank("vi")` pipeline, whose
`VietnameseTokenizer` joins syllables into words with pyvi's CRF model (`use_pyvi = true`),
followed by the sentencizer. This script runs it on fixed inputs and writes what it returns
next to it, together with the steps in between; the replay tests feed every row to the Rust
code and require identical results.

Run inside the pyroid-dev container (never on the host) with an interpreter that has
datatrove==0.10.1, spacy==3.8.18, pyvi==0.1.1 (with scikit-learn, sklearn-crfsuite==0.5.0 and
python-crfsuite==0.9.12) and pyarrow:

    python -I oracle.py record [--out DIR] [--fineweb PARQUET] [--full] [--reverse] [--jobs N]
    python -I oracle.py large PARQUET --out DIR [--reverse] [--jobs N]
    python -I oracle.py compare DIR1 DIR2

Without `--out` the files are written next to this script; `--full` (which needs `--out`)
records every row instead of the committed selection. `--reverse` processes the rows in
reverse order and writes them in the usual order: the output must not change (`compare`
checks that two directories hold identical files). `--jobs` records in that many processes
(the output does not depend on it). `large` records every document of a FineWeb-2 parquet
file into `large.jsonl`. Every run gives the same bytes.

Files, one text per row:

    cases.jsonl     handwritten and generated inputs for every branch of the pipeline: each
                    alternative of the syllable regex and their order (abbreviations, specials,
                    URLs, emails and their backtracking, numbers, punctuation, words), every
                    whitespace kind, the conditions that keep two syllables apart, the model's
                    features (case, digits, lowercasing, n-grams of words.txt), NUL characters,
                    NFD text, sentence ends, empty and very long texts
    fineweb.jsonl   whole FineWeb-2 documents
    fuzz.jsonl      seeded random strings
    sweep.jsonl     code point sweep: every 32 consecutive code points ("run"), and each code
                    point in a fixed context ("ctx": "Hà{c}Nội {c} Việt{c}nam {c}." for 32 code
                    points, joined by "\\n"); surrogates left out

The committed files hold a selection: rows of at most 16 KiB, a seeded sample of the FineWeb-2
documents and of the sweep. `--full` writes every row (the whole sweep is about 70,000 rows).

Every JSONL file starts with a header line `{"header": {...}}`: the level, the generator and
seed, the selection, the versions of datatrove, spaCy, pyvi, python-crfsuite,
sklearn-crfsuite, scikit-learn, Python and its Unicode database, the sha256 of pyvi's model
pickle and word list, the source parquet file (if any) and the number of rows. Then one row
per line:

    {"t": text, "seg": [len(x) for x in segs], "lab": labels, "tok": vt.pyvi_tokenize(t),
     "w": tok.word_tokenize(t), "s": tok.sent_tokenize(t), "sp": tok.span_tokenize(t)}

where tok is `load_word_tokenizer("vie_Latn")`, vt is spaCy's `VietnameseTokenizer`
(`tok.tokenizer.tokenizer`), segs is `vt.pyvi_sylabelize_with_ws(t)` (every match of its
regex, whitespace runs included: a partition of t), and labels has one character per
non-whitespace seg ("words"), "B" for B_W and "I" for I_W, from
`vi.model.predict([vi.sent2features(words, False)])[0]` with vi the class
`vt.ViTokenizer.ViTokenizer`. "seg" and "lab" are left out when t is empty or t.isspace()
(pyvi_tokenize does not run the model then). "sp" holds the spans as [start, end] pairs. When
a call raises, the row has "x_<key>": "<exception type>: <message>" instead of that key. Rows
can also have "k" (what the input is), "d" (the FineWeb-2 row) or "cp" (the first code point
of a sweep row).

A text ("t", and each string in "tok", "w" and "s") is a JSON string or, when that is much
shorter, a recipe: an array of items that are strings or [string, count] pairs; the text is
the concatenation of the strings, each pair's string repeated count times. For example
[["Hà ", 3], "x"] is "Hà Hà Hà x".

Texts containing "IS_ALPHA" are left out everywhere: spaCy's results on them can depend on
what it tokenized before (datatrove retries a failing call without that string).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import multiprocessing
import platform
import random
import re
import string
import sys
import time
import unicodedata
from collections.abc import Callable, Iterator, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

# Look-alike and invisible characters on purpose.
# ruff: noqa: RUF001

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
MANIFEST = REPO / "bench" / "manifest.json"
LANG = "vie_Latn"
SEED = 20261109
FINEWEB = Path("/data/fineweb2/vie_Latn.parquet")
FINEWEB_DATASET = "HuggingFaceFW/fineweb-2 vie_Latn (ODC-By 1.0)"
N_FUZZ = 2000
N_FINEWEB_DOCS = 28  # committed sample of FineWeb-2 documents
FINEWEB_DOC_BYTES = 3000  # committed documents are at most this long
SWEEP_WIDTH = 32
N_SWEEP_BMP, N_SWEEP_REST = 10, 4  # committed sample of each sweep kind, besides SWEEP_FIXED
MAX_ROW = 16 * 1024  # bytes of a committed row
PACK_MIN = 256  # shorter strings are never written as recipes
SKIP = "IS_ALPHA"
LABELS = {"B_W": "B", "I_W": "I"}

Item = tuple[dict[str, Any], str]
Recorder = Callable[[str], dict[str, Any]]


def log(msg: str) -> None:
    sys.stderr.write(f"[oracle {time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


def utf8len(s: str) -> int:
    return len(s.encode("utf-8"))


def err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# Output


# Raw line and paragraph separators and BOMs are legal in JSON strings, but some readers
# split lines on them: write them escaped.
_ESCAPES = str.maketrans({c: f"\\u{ord(c):04x}" for c in "\x85\u2028\u2029\ufeff"})
_RUN = re.compile(r"(.{1,16}?)\1{3,}", re.DOTALL)


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).translate(_ESCAPES)


def unpack(value: str | list[Any]) -> str:
    """The text a string or recipe stands for."""
    if isinstance(value, str):
        return value
    return "".join(x if isinstance(x, str) else x[0] * x[1] for x in value)


def pack(s: str) -> str | list[Any]:
    """`s`, or a recipe for it when that is at most half as long."""
    if len(s) < PACK_MIN:
        return s
    items: list[Any] = []
    pos = 0
    for m in _RUN.finditer(s):
        if m.end() - m.start() < 32:
            continue
        if m.start() > pos:
            items.append(s[pos : m.start()])
        unit = m.group(1)
        items.append([unit, (m.end() - m.start()) // len(unit)])
        pos = m.end()
    if not items:
        return s
    if pos < len(s):
        items.append(s[pos:])
    if 2 * len(dumps(items)) > len(dumps(s)):
        return s
    if unpack(items) != s:
        raise AssertionError(f"bad recipe for {s[:40]!r}")
    return items


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def pyvi_models() -> Path:
    import pyvi

    return Path(pyvi.__file__).parent / "models"


@functools.cache
def environment() -> dict[str, Any]:
    models = pyvi_models()
    return {
        "datatrove": version("datatrove"),
        "spacy": version("spacy"),
        "pyvi": version("pyvi"),
        "python_crfsuite": version("python-crfsuite"),
        "sklearn_crfsuite": version("sklearn-crfsuite"),
        "scikit_learn": version("scikit-learn"),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "unicode": unicodedata.unidata_version,
        "data_sha256": {
            f"pyvi/models/{name}": sha256_file(models / name) for name in ("pyvi3.pkl", "words.txt")
        },
    }


def header(generator: str, selection: str, **extra: Any) -> dict[str, Any]:
    return {
        "level": LANG,
        "generator": generator,
        "seed": SEED,
        "selection": selection,
        **environment(),
        **extra,
    }


def write_jsonl(path: Path, head: dict[str, Any], lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(dumps({"header": {**head, "rows": len(lines)}}) + "\n")
        f.writelines(line + "\n" for line in lines)
    log(f"wrote {path} ({len(lines)} rows, {path.stat().st_size} bytes)")


def small(lines: Sequence[str]) -> list[str]:
    kept = [line for line in lines if utf8len(line) <= MAX_ROW]
    if len(kept) < len(lines):
        log(f"  {len(lines) - len(kept)} rows over {MAX_ROW} bytes left out")
    return kept


# --------------------------------------------------------------------------
# Recording


def put(row: dict[str, Any], key: str, call: Callable[[], Any]) -> None:
    try:
        row[key] = call()
    except Exception as e:
        row[f"x_{key}"] = err(e)


def make_recorder() -> Recorder:
    from datatrove.utils.word_tokenizers import load_word_tokenizer

    tok = load_word_tokenizer(LANG)
    vt = tok.tokenizer.tokenizer  # spaCy's VietnameseTokenizer
    vi = vt.ViTokenizer.ViTokenizer

    def labels(text: str) -> str:
        words = [x for x in vt.pyvi_sylabelize_with_ws(text) if not x.isspace()]
        return "".join(LABELS[x] for x in vi.model.predict([vi.sent2features(words, False)])[0])

    def record(text: str) -> dict[str, Any]:
        row: dict[str, Any] = {}
        if len(text) > 0 and not text.isspace():  # pyvi_tokenize runs the model
            put(row, "seg", lambda: [len(x) for x in vt.pyvi_sylabelize_with_ws(text)])
            put(row, "lab", lambda: labels(text))
        put(row, "tok", lambda: [pack(x) for x in vt.pyvi_tokenize(text)])
        put(row, "w", lambda: [pack(x) for x in tok.word_tokenize(text)])
        put(row, "s", lambda: [pack(x) for x in tok.sent_tokenize(text)])
        put(row, "sp", lambda: [list(x) for x in tok.span_tokenize(text)])
        return row

    return record


_RECORD: Recorder | None = None


def _init_worker() -> None:
    global _RECORD
    _RECORD = make_recorder()


def _record_item(item: Item) -> str:
    assert _RECORD is not None
    meta, text = item
    return dumps({**meta, "t": pack(text), **_RECORD(text)})


class Runner:
    """Records items in this process (jobs=1) or in a pool of `jobs` processes."""

    def __init__(self, jobs: int) -> None:
        self.pool = None
        if jobs > 1:
            ctx = multiprocessing.get_context("spawn")
            self.pool = ctx.Pool(jobs, initializer=_init_worker)
        elif _RECORD is None:
            _init_worker()

    def stream(self, items: Sequence[Item], order: Sequence[int], chunk: int) -> Iterator[str]:
        todo = (items[i] for i in order)
        if self.pool is None:
            return map(_record_item, todo)
        return self.pool.imap(_record_item, todo, chunksize=chunk)

    def lines(self, items: Sequence[Item], reverse: bool, what: str) -> list[str]:
        order = range(len(items) - 1, -1, -1) if reverse else range(len(items))
        lines = [""] * len(items)
        t0 = time.monotonic()
        for done, (i, line) in enumerate(zip(order, self.stream(items, order, 16), strict=True), 1):
            lines[i] = line
            if done % 10_000 == 0:
                log(f"  {what}: {done}/{len(items)} rows, {time.monotonic() - t0:.0f} s")
        log(f"{what}: {len(items)} rows in {time.monotonic() - t0:.1f} s")
        return lines

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


def clean(items: Sequence[Item]) -> list[Item]:
    kept = [it for it in items if SKIP not in it[1]]
    if len(kept) < len(items):
        log(f"  {len(items) - len(kept)} texts containing {SKIP!r} left out")
    return kept


# --------------------------------------------------------------------------
# Inputs: handwritten and generated cases

WHITESPACE = [
    " ", "\t", "\n", "\r\n", "\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x1f", "\x85",
    "\u00a0", "\u1680", "\u2000", "\u2001", "\u2002", "\u2003", "\u2004", "\u2005", "\u2006",
    "\u2007", "\u2008", "\u2009", "\u200a", "\u2028", "\u2029", "\u202f", "\u205f", "\u3000",
]  # fmt: skip
# Not whitespace for str.isspace() and the regex's \s, although they look like it.
NOT_SPACE = ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u180e", "\u00ad", "\x00", "\x7f"]

ABBR_CASES = [
    "A.", "AB.", "ABC.", "Đ.", "ĐĐ.", "ĐHQG.", "TP.", "TP.HCM", "Tp.", "Tp.HCM", "tp.", "TP",
    "Tp", "Mr.", "Mrs.", "Ms.", "Dr.", "ThS.", "Mr", "Mrs", "Ms", "Dr", "ThS", "MR.", "MRS.",
    "MS.", "DR.", "THS.", "mr.", "mrs.", "dr.", "ths.", "Mr.Smith", "Mrs.Smith", "Ms.X",
    "Dr.Ng", "ThS.BS", "TS.", "GS.TS.", "PGS.TS.", "Th.S.", "ThS..", "A.B.C.", "A..", "A...",
    "U.S.A.", "À.", "ÀB.", "AÀ.", "Ă.", "Â.", "Ê.", "Ô.", "Ơ.", "Ư.", "Á.", "Ạ.", "đ.", "AĐ.",
    "ĐA.", "A1.", "1A.", "AB1.", "A_.", "_A.", "Ｍｒ.", "Ａ.", "Ω.", "Σ.", "Ж.", "İ.", "ǅ.",
    "ẞ.", "Tp.Hồ Chí Minh", "ông Mr. A", "bà Mrs. B", "ThS.Nguyễn", "Mrs.Mr.Ms.", "MrMrs.",
    "Tp.Tp.", "aMr.", "aA.", "a.A.", "Ab.", "ThSs.", "Mrss.", "Đđ.", "DR.Smith", "A.B",
]  # fmt: skip

SPECIAL_CASES = [
    "==>", "->", "...", ">>", "=>", "-->", "--->", "===>", "==>>", "==>>>", "->>", ">>>",
    ">>>>", "....", ".....", "......", "<-", "<<", "<==", "=", "==", ">", "-", "a==>b", "a->b",
    "a...b", "a>>b", "x ==> y", "1->2", "1...2", "1.2...3", "...1.2", "…", "……", "a…b", "– —",
    "-->>", ".-.", "-.", "..", ". . .", "\u2024\u2025\u2026", "⇒", "→", "=>>", "-=>", "=->",
    ">->", "->->", "...->", "->...", "Tp...", "A...", "Mr...", "1..", "1...", "a.->b",
]  # fmt: skip

WEB_CASES = [
    "http://example.com", "https://vnexpress.net/thoi-su/abc-123.html",
    "https://vi.wikipedia.org/wiki/Hà_Nội", "ftp://a.b/c", "http://", "http:// x", "http://x",
    "a://b", "1://2", "_://x", "Hà://Nội", "http://a\u00a0b", "http://a\u3000b",
    "http://a\u200bb", "http://a\x00b", "http://a\tb", "xhttp://y", "(http://x.y)",
    "http://x.y.", "http://x.y, z", '"http://x"', "http://x/ http://y", "mailto:a@b.c",
    "http://a@b.c", "a@b.c://d", "://x", "x:/y", "x:///y", "http:///", "https://xn--fsq.vn",
    "HTTP://X.Y", "ǅ://x", "²://x", "١://x", "http://ă.vn/đ?q=ư#ơ", "http://x\x85y",
    "http://x\u2028y", "Tp.://x", "Mr.http://x", "...http://x", "http://x...", "->http://x",
    "a-b://c", "a.b://c", "http:\\\\x", "http//x", "https:/x", "ftp://\x00",
]  # fmt: skip

EMAIL_CASES = [
    "a@b.c", "a@b.c.", "a@b.c.d", "a@b..c", "a@.b", "a@b", "@b.c", "a@", "@", "a.b@c.d",
    "a..b@c.d", ".a@b.c", "a.@b.c", "+a@b.c", "-a@b.c", "_a@b.c", "a+tag@b.c", "a_b@c-d.e",
    "a@b-.c", "a@-b.c", "a@b.c-", "a@b.c_d", "a@b.c.d.e.f", "a@b.c..d", "a@@b.c", "a@b@c.d",
    "é@a.b", "a@é.b", "a@b.é", "a@b.cé", "A@B.C", "1@2.3", "1.2@3.4", "1.2.3@4", "x@y.z,",
    "(a@b.c)", "<a@b.c>", "a@b.c/d", "ABC.x@y.z", "Mr.x@y.z", "Tp.x@y.z", "...a@b.c",
    "a@b.c...", "->a@b.c", "nguyen.van.a@gmail.com", "info@vnexpress.net.",
    "Liên hệ: hotro@abc.com.vn.", "a@b.c\x00d", "a@b.c\u00a0d", "ａ@ｂ.ｃ", "a@b.１", "a@1.2",
    "a@b.c.1", "a@b._", "a@b.-", "a@b.c.-", "a-@b.c", "a@b.c@d.e", "a@b.c.d@e.f", "a.b.c@d",
    "A.B.@c.d", "AB.@c.d", "a @b.c", "a@ b.c", "a@b .c", "Đ@b.c", "a@b.Đ", "x.y@z.w.",
    "1.2.3.4@5.6", "1,2@3.4", "1_2@3.4", "a@b.c_", "_@_.__", "-@-.-", "+@a.b", ".@a.b",
]  # fmt: skip

DIGIT_CASES = [
    "1", "12", "1.2", "1,2", "1_2", "1.2.3", "1,234,567", "1.234.567", "1.234,56", "3.14", "1.",
    "1.2.", "1..2", "1.,2", ".5", ",5", "_5", "1_", "1_a", "1_2_a", "1.2a", "a1.2", "1.2.3a",
    "12abc", "abc12", "1a.2", "2020", "2.020.000", "0.5%", "50%", "1/2", "1:30", "12-34",
    "1-2-3", "10^3", "1e5", "1.2e3", "+84", "+84.123", "-1.5", "1 000", "1 000 000", "1\u202f000",
    "١٢٣", "١٢.٣٤", "١.٢", "๑๒๓", "๑.๒", "١,٢", "１２", "１.２", "１，２", "𝟏.𝟐", "²", "²³",
    "1²", "²1", "①", "①②", "1.①", "Ⅻ", "Ⅻ.Ⅰ", "½", "1½", "一", "一.二", "٣.x",
    "1.2.3.4.5.6.7.8.9.10",
    "0,5", "3,14", "1,2,3", "1,,2", "1__2", "1._2", "x_1.2", "_1.2", "Đ1.2", "A1.2", "1.2.Đ",
    "12h30", "12h", "30/4/1975", "2/9", "10.000đ", "100.000 VNĐ", "1,5 triệu", "50.000 người",
    "3.5 tỷ", "TP.1.2", "Mr.1.2", "1.2_3,4", "1_2.3", "0.0.0.0", "1.2..3.4", "1.2,", "1.2_",
    "٠.٠", "߁.߂", "१.२", "১.২", "໑.໒", "၁.၂", "፩.፪", "𑁧.𑁨", "1.٢", "٢.1", "1.²", "²³.1",
    "Ⅳ.Ⅴ", "1.½", "00.00", "007", "0x1F", "1.2-3.4", "1.2/3.4",
]  # fmt: skip

PUNCT_CASES = [
    *string.punctuation,
    *"“”‘’«»–—…·•°§¶©®™€£¥₫¿¡、。，！？（）《》【】‹›„‚′″",
    "()", "[]", "{}", '""', "''", "?!", "!?", "!!", "??", ".,", ",.", "Hà(Nội)", "(Hà Nội)",
    "Hà - Nội", "Hà-Nội", "Hà–Nội", "Hà—Nội", "Hà_Nội", "Hà _ Nội", "Hà/Nội", "Hà & Nội",
    "Hà's", "Hà'", "'Hà'", '"Hà Nội"', "“Hà Nội”", "«Hà Nội»", "Hà, Nội", "Hà; Nội", "Hà: Nội",
    "Hà.Nội", "Hà...Nội", "Hà…Nội", "#HàNội", "@HàNội", "$100", "100$", "100₫", "100%", "C++",
    "C#", "a&b", "a|b", "a~b", "a`b", "a^b", "a\\b", "a*b", "a+b", "a=b", "a<b>c", "x ° y",
    "Ⓐ Ⓑ", "ⓐ", "☺", "😀", "Hà 😀 Nội", "👍🏽", "🇻🇳", "👨\u200d👩\u200d👧", "Hà – Nội",
    "thủ – đô", "thủ … đô", "thủ “ đô", "thủ · đô", "thủ ° đô", "thủ ☺ đô", "thủ - đô",
    "thủ _ đô", "thủ ( đô", "thủ ) đô", "thủ \x00 đô", "thủ ² đô", "thủ ① đô",
]  # fmt: skip

CASE_CASES = [
    "ǅ", "ǅa", "ǄA", "ǆ", "Dž", "ǅungla", "İ", "İstanbul", "İİ", "i\u0307", "ΑΣ", "ΣΑΣ",
    "ΑΣ ΑΣ", "Σ", "σς", "ΑΣ1", "ΑΣa", "ΟΔΥΣΣΕΥΣ", "ΑΣ'Α", "ẞ", "ẞA", "ß", "SS", "Ꭰ", "ᎠᎡᎢ",
    "ꭰ", "Ꭰꭰ", "Ა", "ᲐᲑᲒ", "ა", "აბგ", "Ⴀ", "ⴀ", "Ⓐ", "ⓐ", "Ⅻ", "ⅻ", "ʰ", "ᾼ", "ᾳ", "ᾈ", "ﬀ",
    "ﬁ", "Ǉ", "ǈ", "ǉ", "ǈa", "Ω", "\u2126", "\u212a", "\u212b", "ı", "I", "ŉ", "ǰ", "ΐ", "ﬃ",
    "Hà", "HÀ", "hà", "hÀ", "HàNội", "HÀNỘI", "Hà1", "1Hà", "_Hà", "Hà_Nội", "HÀ_NỘI",
    "VIỆT NAM", "Việt Nam", "việt nam", "vIệT nAm", "Việt nam", "việt Nam", "ĐÀ NẴNG",
    "Đà Nẵng", "đà nẵng", "Đà nẵng", "đà Nẵng", "ĐàNẵng", "Đ", "đ", "Ð", "ð", "Ɖ", "ɖ", "𝐀",
    "𝐀𝐁", "Ａ", "ａ", "ＡＢ", "Ａｂ", "Ǳ", "ǲ", "ǳ", "ǲa b", "ᏸ", "Ᏸ", "ꮰ", "Ꭰa", "𐐀", "𐐨",
    "𞤀", "𞤢", "ꞵ", "Ꞵ", "ǅ1", "1ǅ", "A1", "1A", "_A", "A_", "AB_cd", "Ab_Cd", "ⅰⅱ", "ⅠⅡ",
]  # fmt: skip

SENT_CASES = [
    "Câu một. Câu hai.", "Câu một.Câu hai", "Câu một! Câu hai? Câu ba.", "Câu một… Câu hai",
    "Câu một... Câu hai", "Câu một.. Câu hai", "Câu một.... Câu hai", "Câu một..... Câu hai",
    "Câu một。Câu hai", "Câu một｡ Câu hai", "Câu một！Câu hai？Câu ba．", "Câu một?! Câu hai",
    "Câu một !? Câu hai", 'Câu một." Câu hai', "Câu một.” Câu hai", "Câu một.) Câu hai",
    "Câu một.] Câu hai", "Câu một.»  Câu hai", "Câu một.' Câu hai", 'Câu một.") Câu hai',
    "Câu một. - Câu hai", "Câu một. – Câu hai", "Câu một. 1. Câu hai", "1. Mục một 2. Mục hai",
    "Câu một.\nCâu hai", "Câu một.\n\nCâu hai", "Câu một.\r\nCâu hai", "Câu một.  Câu hai",
    "Câu một.\tCâu hai", "Câu một.\u00a0Câu hai", "Câu một.\u3000Câu hai", "Câu một .", ". Câu",
    ".", "..", "...", "!", "?", "?!", ". . .", "Câu một. ", " Câu một.", "  Câu một.  ",
    "Câu một.\n", "\nCâu một.", "Ông A. Nguyễn", "TP. Hồ Chí Minh", "Tp. Hồ Chí Minh",
    "Mr. Smith đến.", "Dr. A và ThS. B.", "Năm 1.2. Tiếp", "Câu ‼ Câu", "Câu ‽ Câu", "Câu ⁇ Câu",
    "Câu ։ Câu", "Câu ؟ Câu", "Câu । Câu", "Câu ። Câu", "Câu ﹒ Câu", "Câu 𑁇 Câu", "Câu ꓿ Câu",
    "Câu.\u200bCâu", "Câu.😀 Câu", "Câu. 😀 Câu", "Câu.\x00Câu", "Câu.…Câu", "a.b", "a. b",
    "a .b", "Hà Nội. Việt Nam.", "Thủ đô Hà Nội. Việt Nam!", "Câu.\n\n\nCâu", "Câu. \n Câu",
    "Câu.\x0bCâu", "Câu.\x0cCâu", "Câu.\x1cCâu", "Câu.\x85Câu", "Câu.\u2028Câu",
    "Câu. . Câu", "Câu .. Câu", "Câu ... . Câu", "Câu ! ! Câu", "Câu.!? Câu", "Câu?.. Câu",
    "Câu. “Câu hai.” Câu ba.", "(Câu một.) Câu hai.", "Câu một.Tp. Hồ", "a.b.c.d", "A. B. C.",
    "Hà Nội.Việt Nam", "thủ đô. Hà Nội", "x.y@z.w. Câu", "http://x.y. Câu", "1.2. 3.4.",
]  # fmt: skip

NUL_CASES = [
    "\x00", "\x00\x00", "a\x00b", "Hà\x00Nội", "Hà \x00 Nội", "http://a\x00b",
    "http://x\x00 y", "\x00Hà Nội", "Hà Nội\x00", "a\x00@b.c", "\x00.", ".\x00", "1\x00.2",
    "Mr.\x00", "Hà\x00 Nội", "http://a\x00b Hà Nội", "Việt http://x\x00y Nam", "thủ đô\x00",
    "\x00thủ đô", "thủ\x00đô", "http://\x00hà nội", "http://hà\x00 nội", "\x00\x01\x02\x03",
    "\x7f", "Hà\x7fNội", "\x1b[31mđỏ\x1b[0m",
]  # fmt: skip

# The model labels the second syllable I_W, and a single condition keeps the two apart: the
# previous or the current one is in string.punctuation, starts with a digit, or is title case
# after one that is not.
MERGE_CASES = [
    "cánh ! nửa", "cánh!nửa", "cánh! nửa", "sinh ! thiết", "sinh!thiết", "# nhập", "# âm",
    "nghi #vệ", "nghi# vệ", "bất $ cẩn", "bất$cẩn", "bất$ cẩn", "tổng * kho", "tổng*kho",
    "di *bút", "di* bút", "thủ Đô", "hà Nội", "việt Nam", "Hà nội", "HÀ nội", "thủ ǅô",
    "năm 2020", "lần 2 thứ", "thứ 2", "2 người", "² người", "① người", "người ²", "người ๑",
]  # fmt: skip

MIXED_CASES = [
    "Hà Nội là thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam.",
    "Thành phố Hồ Chí Minh (TP.HCM) có hơn 9 triệu dân.",
    "Ông Nguyễn Văn A, Giám đốc Sở Y tế, cho biết: “Tình hình đã được kiểm soát.”",
    "Giá vàng hôm nay tăng 1,5% lên 85.000.000 đồng/lượng.",
    "Liên hệ: hotro@congty.com.vn hoặc truy cập https://congty.vn/lien-he.",
    "Ngày 2/9/1945, Chủ tịch Hồ Chí Minh đọc bản Tuyên ngôn Độc lập.",
    "ThS. Trần Thị B và Dr. Lê Văn C đã tham dự hội thảo.",
    "Mr. Smith và Mrs. Smith đến Việt Nam vào năm 2019.",
    "Đội tuyển Việt Nam thắng 3-0 trước Thái Lan!",
    "Bạn có khỏe không? Tôi khỏe, cảm ơn.",
    "Tiếng Việt có 6 thanh điệu: ngang, huyền, sắc, hỏi, ngã, nặng.",
    "Sản phẩm được giảm giá 50%... Nhanh tay lên!!!",
    "Covid-19 đã ảnh hưởng đến kinh tế toàn cầu.",
    "Điện thoại: 0912.345.678 - Fax: (028) 3823 4567",
    "Học sinh lớp 12A1 trường THPT Chu Văn An ==> đạt giải nhất.",
    "Bước 1 -> Bước 2 -> Bước 3 >> Hoàn thành",
    "UBND TP. Đà Nẵng vừa ban hành quyết định số 123/QĐ-UBND.",
    "Công ty TNHH ABC có trụ sở tại Q.1, TP.HCM.",
    "Chương trình “Mùa xuân cho em” năm nay đã quyên góp được 1,2 tỷ đồng.",
    'Anh ấy nói: "Tôi sẽ về quê vào Tết Nguyên Đán."',
    "Nhiệt độ cao nhất là 38°C, thấp nhất 27°C.",
    "Theo GS.TS Nguyễn Văn D, Đại học Quốc gia Hà Nội (ĐHQGHN)...",
    "#VietNam #HaNoi @user123",
    "Từ khóa: học máy, trí tuệ nhân tạo, xử lý ngôn ngữ tự nhiên",
    "VIỆT NAM VÔ ĐỊCH!",
    "việt nam vô địch",
    "Xin chào, tôi là sinh viên năm thứ 3 của trường Đại học Bách khoa.",
    "Cà phê sữa đá là thức uống phổ biến ở Sài Gòn.",
    "Lần thứ 2 tôi đến Hà Nội là vào năm 2010, khi tôi 25 tuổi.",
    "Hôm nay là thứ Hai, ngày 12 tháng 3 năm 2024.",
    "Ông ấy là người Hà Nội nhưng sống ở thành phố Hồ Chí Minh.",
    "Tôi yêu Việt Nam. Tôi yêu Hà Nội. Tôi yêu tiếng Việt.",
    "Nhà xuất bản Giáo dục Việt Nam, 2020, tr. 15-20.",
    "Email: abc.def@gmail.com; Website: http://www.abc.vn; ĐT: 024.3824.5678",
    "Thời gian: 8h00 - 17h30, từ thứ 2 đến thứ 6.",
    "Chiều dài 1.234,5 m; chiều rộng 2,5 m; diện tích 3.086,25 m².",
    "Q: Bạn tên gì? A: Tôi tên là Lan.",
    "Đường Lê Lợi, P. Bến Thành, Q. 1, TP. HCM",
    "Truyện Kiều của Nguyễn Du gồm 3.254 câu thơ lục bát.",
]

RUNS = [
    "a", "Hà", "Hà Nội ", "Hà_", "1", "1.", "1.2", ".", "!", "-", "->", "==>", "a@b.", "Mr.",
    "Tp.", "A.", " ", "\n", "Hà\nNội\n", "việt nam ", "VIỆT NAM ", "ǅ", "\x00", "😀", "a\u0301",
    "Câu một. ", "thủ đô ", "Thủ Đô ", "1 ", "(", "…", "\u00a0", "Hà\u00a0", "a b ", "ΑΣ ",
    "http://x ", "_", "²", "Đ.",
]  # fmt: skip


def case_variants(x: str) -> list[str]:
    """`x` alone and between Vietnamese words."""
    return [x, f"{x} nội", f"hà {x}", f"thủ đô{x}"]


def ngram_items(rng: random.Random) -> list[str]:
    """Texts around bigrams and trigrams of pyvi's words.txt, the model's n-gram features."""
    entries = (pyvi_models() / "words.txt").read_text(encoding="utf-8").split("\n")
    two = sorted({e for e in entries if len(e.split(" ")) == 2 and e == e.lower()})
    three = sorted({e for e in entries if len(e.split(" ")) == 3 and e == e.lower()})
    cased = sorted({e for e in entries if e != e.lower() and len(e.split(" ")) in (2, 3)})
    out: list[str] = []
    for e in rng.sample(two, 20):
        a, b = e.split(" ")
        out += [e, f"{a.title()} {b.title()}", e.upper(), f"{a} {b.title()}", f"{a.title()} {b}"]
        out += [f"{a}  {b}", f"{a}\t{b}", f"{a}\n{b}", f"{a}\u00a0{b}", f"{a}, {b}", f"{a}-{b}"]
        out += [f"{a}_{b}", f"{a}. {b}", f"Tôi thấy {e} rồi.", f"{e} {e}", f"{a} {b}{a} {b}"]
        out += [f"{a} 2{b}", f"2{a} {b}", f"{a} ²{b}", f"{a} {b[0].upper()}{b[1:]}"]
    for e in rng.sample(three, 14):
        a, b, c = e.split(" ")
        out += [e, e.title(), e.upper(), f"{a} {b.title()} {c}", f"{a}  {b}\n{c}"]
        out += [f"{a} {b}", f"{b} {c}", f"Ở {e} có {e}.", f"{a}, {b} {c}", f"{a} {b}. {c}"]
    for e in rng.sample(cased, 10):
        out += [e, e.lower(), e.upper()]
    chain = " ".join(rng.sample(two, 12))
    out += [chain, chain.title(), chain.upper(), chain.replace(" ", "")]
    return out


def case_items(fineweb_texts: Sequence[str]) -> list[Item]:
    rng = random.Random(f"{SEED}/cases")
    out: list[Item] = []

    def add(kind: str, *texts: str) -> None:
        out.extend(({"k": kind}, t) for t in texts)

    add("empty", "")
    for w in WHITESPACE + NOT_SPACE:
        add("ws", w, w * 3, f"Hà{w}Nội", f"a{w}b", f"{w}Hà Nội", f"Hà Nội{w}", f"thủ{w}{w}đô")
        add("ws", f"{w}a{w}", f"Câu.{w}Câu", f"{w}.{w}", f"1{w}2", f"http://x{w}y", f"Mr.{w}A")
    add("ws", "".join(WHITESPACE), " \t\n\r\x0b\x0c", " \u00a0 ", "\u00a0\u00a0", "\r\n\r\n")
    add("ws", " \n ", "\n \n", "a \t b", "Hà \u3000 Nội", "  Hà  Nội  ", "\u2028\u2029")
    add("abbr", *ABBR_CASES)
    add("abbr", *(f"Ông {x} Nguyễn" for x in ABBR_CASES[::3]))
    add("special", *SPECIAL_CASES)
    add("special", *(f"Hà {x} Nội" for x in SPECIAL_CASES[:12]))
    add("web", *WEB_CASES)
    add("email", *EMAIL_CASES)
    add("email", *(f"Gửi {x} nhé" for x in EMAIL_CASES[::2]))
    add("digit", *DIGIT_CASES)
    add("digit", *(f"năm {x} người" for x in DIGIT_CASES[::2]))
    add("punct", *PUNCT_CASES)
    for x in CASE_CASES:
        add("case", *case_variants(x))
    add("ngram", *ngram_items(rng))
    add("sent", *SENT_CASES)
    add("nul", *NUL_CASES)
    add("merge", *MERGE_CASES)
    add("mixed", *MIXED_CASES)
    nfd = [unicodedata.normalize("NFD", t) for t in MIXED_CASES[:16]]
    nfkd = [unicodedata.normalize("NFKD", t) for t in MIXED_CASES[16:24]]
    add("nfd", *nfd, *nfkd, "\u0301", "\u0301\u0301", "a\u0301\u0301", "\u0300Hà", "Ha\u0300")
    add(
        "nfd", "Vie\u0302\u0323t Nam", "Vi\u1ec7t Nam", "Vie\u0323\u0302t Nam", "Ê\u0301", "ê\u0301"
    )
    add("nfd", "thu\u0309 đô", "thủ\u0300 đô", "\u0323", "ơ\u0309", "o\u031b\u0309")
    for r in RUNS:
        add("long", r * max(1, 120 // len(r)), r * max(1, 10_000 // len(r)))
    add("long", "a" * 100_000, "Hà Nội " * 15_000, "x" * 4000 + "@" + "y" * 4000 + ".z")
    add("long", "http://" + "x" * 50_000, "1." * 20_000 + "1", "Tp." * 3000)
    for t in MIXED_CASES[:4]:
        add("long", " ".join([t] * 300))

    # FineWeb-2 lines in odd contexts.
    lines = [x for t in fineweb_texts[:40] for x in t.splitlines() if 8 <= len(x) <= 120]
    for line in rng.sample(lines, min(16, len(lines))):
        add("fw-ctx", f"  {line}  ", unicodedata.normalize("NFD", line), line.upper())
        add("fw-ctx", line.lower(), line.replace(" ", "\u00a0"), line.replace(". ", "."))
    return clean(out)


# --------------------------------------------------------------------------
# Inputs: FineWeb-2, fuzz, sweep


def read_fineweb(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=["text"]).column("text").to_pylist()


def fineweb_source(path: Path, n: int) -> dict[str, Any]:
    """The parquet file: its sha256 and rows, and where it comes from (bench/manifest.json)."""
    src: dict[str, Any] = {"local": f"{path.parent.name}/{path.name}", "sha256": sha256_file(path)}
    src.update(rows=n, dataset=FINEWEB_DATASET)
    try:
        entry = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"].get(src["local"])
    except (OSError, ValueError, KeyError):
        entry = None
    if entry and entry.get("sha256") == src["sha256"]:
        s = entry["source"]
        src.update(repo=s["repo"], revision=s["revision"], file=s["file"])
    return src


def doc_items(docs: Sequence[str]) -> list[Item]:
    return clean([({"d": d}, doc) for d, doc in enumerate(docs)])


def sample_docs(items: Sequence[Item]) -> list[Item]:
    rng = random.Random(f"{SEED}/fineweb-docs")
    fits = [i for i, (_, text) in enumerate(items) if utf8len(text) <= FINEWEB_DOC_BYTES]
    return [items[i] for i in sorted(rng.sample(fits, min(N_FINEWEB_DOCS, len(fits))))]


# Blocks the committed sweep always has: controls and whitespace, ASCII, Latin-1, the
# Vietnamese letters, combining marks, Greek, general punctuation, digits of other scripts.
SWEEP_FIXED = [
    0x0, 0x20, 0x40, 0x60, 0x80, 0xC0, 0x1C0, 0x300, 0x3A0, 0x660, 0x1EA0, 0x1EC0, 0x1EE0,
    0x2000, 0x2460, 0x3000, 0xFF00, 0x1D7C0,
]  # fmt: skip


def sweep_items(sample: bool) -> list[Item]:
    bases = [b for b in range(0, 0x110000, SWEEP_WIDTH) if not 0xD800 <= b < 0xE000]
    kinds = {}
    for kind in ("run", "ctx"):
        if sample:
            rng = random.Random(f"{SEED}/sweep-{kind}")
            bmp = [b for b in bases if b < 0x10000 and b not in SWEEP_FIXED]
            rest = [b for b in bases if b >= 0x10000 and b not in SWEEP_FIXED]
            picked = rng.sample(bmp, N_SWEEP_BMP) + rng.sample(rest, N_SWEEP_REST)
            kinds[kind] = sorted(picked + SWEEP_FIXED)
        else:
            kinds[kind] = bases
    items: list[Item] = [
        ({"k": "run", "cp": b}, "".join(map(chr, range(b, b + SWEEP_WIDTH)))) for b in kinds["run"]
    ]
    items += [
        (
            {"k": "ctx", "cp": b},
            "\n".join(
                f"Hà{chr(c)}Nội {chr(c)} Việt{chr(c)}nam {chr(c)}." for c in range(b, b + 32)
            ),
        )
        for b in kinds["ctx"]
    ]
    return clean(items)


def vietnamese_letters() -> tuple[list[str], list[str]]:
    """Every precomposed Vietnamese letter, lowercase and uppercase."""
    base = "aăâeêioôơuưy"
    tones = ["", "\u0300", "\u0301", "\u0309", "\u0303", "\u0323"]
    vowels = [unicodedata.normalize("NFC", v + t) for v in base for t in tones]
    lower = sorted(set(vowels + list("bcdđghklmnpqrstvx")))
    return lower, [c.upper() for c in lower]


def fuzz_items(fineweb_texts: Sequence[str]) -> list[Item]:
    rng = random.Random(f"{SEED}/fuzz")
    lower, upper = vietnamese_letters()
    entries = (pyvi_models() / "words.txt").read_text(encoding="utf-8").split("\n")
    sylls = sorted({s for e in entries if e == e.lower() for s in e.split(" ") if s})
    pools: dict[str, list[str]] = {
        "lower": lower,
        "upper": upper,
        "ascii": [chr(c) for c in range(0x30, 0x7B) if chr(c).isalnum()],
        "punct": list(string.punctuation),
        "special": ["==>", "->", "...", ">>", "://", "@", "Tp.", "Mr.", "Mrs.", "Ms.", "Dr."],
        "abbr": ["ThS.", "A.", "Đ.", "TP.", "http://", "www.", ".com", ".vn", "@gmail.com"],
        "digit": list("0123456789.,_"),
        "ws": WHITESPACE + NOT_SPACE,
        "comb": ["\u0300", "\u0301", "\u0303", "\u0309", "\u0323", "\u0302", "\u0306", "\u031b"],
        "case": ["ǅ", "İ", "Σ", "ΑΣ", "ẞ", "Ꭰ", "Ა", "ა", "Ⓐ", "ʰ", "ᾼ", "ǈ", "ß", "ı"],
        "digits": ["²", "①", "๑", "١", "٣", "１", "𝟏", "Ⅻ", "½", "一", "߁"],
        "nul": ["\x00", "\x01", "\x7f", "\x1b"],
        "emoji": ["😀", "👍", "🇻🇳", "👨\u200d👩\u200d👧", "❤\ufe0f", "1\ufe0f\u20e3"],
        "uni": list("“”‘’«»–—…·•°€₫、。！？"),
    }
    names = [*pools, "syll", "syll", "syll", "Syll", "fineweb", "fineweb"]
    items: list[Item] = []
    while len(items) < N_FUZZ:
        pieces = []
        for _ in range(rng.randint(1, 8)):
            name = rng.choice(names)
            if name == "fineweb":
                doc = rng.choice(fineweb_texts)
                at = rng.randrange(len(doc))
                piece = doc[at : at + rng.randint(1, 16)]
            elif name in ("syll", "Syll"):
                piece = rng.choice(sylls)
                piece = piece.title() if name == "Syll" else piece
            else:
                piece = "".join(rng.choice(pools[name]) for _ in range(rng.randint(1, 3)))
            if rng.random() < 0.06:
                piece *= rng.randint(2, 4)
            pieces.append(piece)
            pieces.append(rng.choice([" ", " ", " ", "", "", "\n", ", ", ". "]))
        text = "".join(pieces)[: rng.choice([10, 24, 40])]
        if SKIP not in text:
            items.append(({}, text))
    return items


# --------------------------------------------------------------------------
# Commands


def out_dir(args: argparse.Namespace) -> Path:
    out: Path = args.out
    if getattr(args, "full", False) and out.resolve() == HERE:
        sys.exit("--full writes large files: give an --out directory outside the repository")
    return out


def cmd_record(args: argparse.Namespace) -> None:
    out = out_dir(args)
    full = args.full
    selection = "full" if full else "committed"
    docs = read_fineweb(args.fineweb)
    src = fineweb_source(args.fineweb, len(docs))
    runner = Runner(args.jobs)

    def emit(generator: str, items: Sequence[Item], **extra: Any) -> None:
        lines = runner.lines(items, args.reverse, generator)
        if not full:
            lines = small(lines)
        write_jsonl(out / f"{generator}.jsonl", header(generator, selection, **extra), lines)

    emit("cases", case_items(docs), source=src)
    whole = doc_items(docs)
    emit("fineweb", whole if full else sample_docs(whole), source=src)
    emit("fuzz", fuzz_items(docs), source=src)
    emit("sweep", sweep_items(sample=not full))
    runner.close()


def cmd_large(args: argparse.Namespace) -> None:
    out = out_dir(args)
    docs = read_fineweb(args.parquet)
    src = fineweb_source(args.parquet, len(docs))
    items = doc_items(docs)
    chars = sum(len(t) for _, t in items)
    log(f"{args.parquet}: {len(items)} documents, {chars} characters")
    head = header("large", "full", source=src, characters=chars)
    runner = Runner(args.jobs)
    t0 = time.monotonic()
    if args.reverse:
        write_jsonl(out / "large.jsonl", head, runner.lines(items, True, "large"))
    else:
        path = out / "large.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(dumps({"header": {**head, "rows": len(items)}}) + "\n")
            for done, line in enumerate(runner.stream(items, range(len(items)), 4), 1):
                f.write(line + "\n")
                if done % 1000 == 0:
                    log(f"  large: {done}/{len(items)} rows, {time.monotonic() - t0:.0f} s")
        log(f"wrote {path} ({len(items)} rows, {path.stat().st_size} bytes)")
    runner.close()
    log(f"large: {len(items)} rows, {chars} characters, {time.monotonic() - t0:.1f} s")


def cmd_compare(args: argparse.Namespace) -> None:
    a: Path = args.a
    b: Path = args.b
    names = sorted(p.name for p in a.iterdir() if p.suffix in (".json", ".jsonl"))
    bad = 0
    for name in names:
        if not (b / name).exists():
            log(f"{name}: missing from {b}")
            bad += 1
            continue
        la = (a / name).read_text(encoding="utf-8").split("\n")
        lb = (b / name).read_text(encoding="utf-8").split("\n")
        if la == lb:
            log(f"{name}: identical ({len(la) - 1} lines)")
            continue
        bad += 1
        first = next(
            (i for i, (x, y) in enumerate(zip(la, lb, strict=False)) if x != y),
            min(len(la), len(lb)),
        )
        log(f"{name}: differs, first at line {first + 1} ({len(la)} vs {len(lb)} lines)")
    sys.exit(1 if bad else 0)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("record", help="cases, FineWeb-2 documents, fuzz and the code point sweep")
    s.add_argument("--out", type=Path, default=HERE)
    s.add_argument("--fineweb", type=Path, default=FINEWEB)
    s.add_argument("--full", action="store_true", help="every row, not the committed ones")
    s.add_argument("--reverse", action="store_true", help="process rows in reverse order")
    s.add_argument("--jobs", type=int, default=1, help="processes (default 1)")
    s = sub.add_parser("large", help="every document of a FineWeb-2 parquet file")
    s.add_argument("parquet", type=Path)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--reverse", action="store_true", help="process rows in reverse order")
    s.add_argument("--jobs", type=int, default=1, help="processes (default 1)")
    s = sub.add_parser("compare", help="check that two directories hold identical recordings")
    s.add_argument("a", type=Path)
    s.add_argument("b", type=Path)
    args = p.parse_args(argv)
    commands: dict[str, Callable[[argparse.Namespace], None]] = {
        "record": cmd_record,
        "large": cmd_large,
        "compare": cmd_compare,
    }
    commands[args.cmd](args)


if __name__ == "__main__":
    main()
