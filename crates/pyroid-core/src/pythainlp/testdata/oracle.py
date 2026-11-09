"""Record what PyThaiNLP's Thai tokenizers and datatrove's Thai word tokenizer return.

The goldens record the output of PyThaiNLP 5.4.0 (Apache-2.0), © 2016-2026 PyThaiNLP Project,
with its word list and CRFCut model, run in python-crfsuite 0.9.12, and of datatrove 0.10.1
(Apache-2.0), © HuggingFace. The FineWeb-2 excerpts are from HuggingFaceFW/fineweb-2
(ODC-By 1.0), subset tha_Thai.

pyroid's Thai word tokenizer must give exactly what datatrove's
`load_word_tokenizer("tha_Thai")` (`ThaiTokenizer`) gives: PyThaiNLP's newmm in safe mode for
words, and CRFCut for sentences. This script runs it on fixed inputs and writes what it
returns next to it, together with the steps in between; the replay tests feed every row to
the Rust code and require identical results.

Run inside the pyroid-dev container (never on the host) with an interpreter that has
datatrove==0.10.1, pythainlp==5.4.0, python-crfsuite==0.9.12 and pyarrow:

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

    cases.jsonl     handwritten and generated inputs for every branch: each alternative of the
                    TCC regex (and its lookahead before a final newline), every character of the
                    Thai block, dictionary words and ambiguity, newmm's graph cutoff,
                    out-of-dictionary runs, the non-Thai patterns, safe mode's chunking,
                    rejoin_formatted_num, CRFCut's enders, starters and sentence-end rules,
                    whitespace kinds, NUL, "|" and "=" in tokens, emoji, combining marks
    fineweb.jsonl   whole FineWeb-2 documents
    fuzz.jsonl      seeded random strings
    sweep.jsonl     code point sweep: every 32 consecutive code points ("run"), and each code
                    point in a fixed context ("ctx": "ก{c}ขา {c}ไทย{c}" for 32 code points,
                    joined by "\\n"); surrogates left out
    long.jsonl      texts of 140 characters to 60 kB (FineWeb-2 documents joined, text without
                    whitespace, synthetic runs): safe mode's chunks and long CRFCut inputs

The committed files hold a selection: rows of at most 16 KiB, a seeded sample of the FineWeb-2
documents and of the sweep. `--full` writes every row (the whole sweep is about 70,000 rows).

Every JSONL file starts with a header line `{"header": {...}}`: the level, the generator and
seed, the selection, the versions of datatrove, PyThaiNLP, python-crfsuite, Python and its
Unicode database, the sha256 of the word list and the CRFCut model, the source parquet file
(if any) and the number of rows. Then one row per line:

    {"t": text, "tcc": [len(x) for x in tcc_p.tcc(t)],
     "nm": [len(x) for x in newmm.segment(t)],
     "nms": [len(x) for x in newmm.segment(t, safe_mode=True)],
     "toks": [len(x) for x in toks], "lab": "".join(crfcut._tagger.tag(features)),
     "w": tok.word_tokenize(t), "s": tok.sent_tokenize(t), "sp": tok.span_tokenize(t)}

where tcc_p, newmm and crfcut are modules of `pythainlp.tokenize`, tok is
`load_word_tokenizer("tha_Thai")`, toks is `pythainlp.tokenize.word_tokenize(t)` (newmm,
whitespace kept, formatted numbers rejoined: what CRFCut segments) and features is
`crfcut._extract_features(toks)`. "lab" holds CRFCut's raw labels ("I" or "E" per token),
before its rules for the last token, terminal punctuation and whitespace; it is left out when
toks is empty. "sp" holds the spans as [start, end] pairs. When a call raises, the row has
"x_<key>": "<exception type>: <message>" instead of that key. Rows can also have "k" (what the
input is), "d" (the FineWeb-2 row) or "cp" (the first code point of a sweep row).

A text ("t", and each string in "w" and "s") is a JSON string or, when that is much shorter, a
recipe: an array of items that are strings or [string, count] pairs; the text is the
concatenation of the strings, each pair's string repeated count times. For example
[["ไทย ", 3], "x"] is "ไทย ไทย ไทย x".
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
LANG = "tha_Thai"
SEED = 20261109
FINEWEB = Path("/data/fineweb2/tha_Thai.parquet")
FINEWEB_DATASET = "HuggingFaceFW/fineweb-2 tha_Thai (ODC-By 1.0)"
N_FUZZ = 1500
N_FINEWEB_DOCS = 24  # committed sample of FineWeb-2 documents
FINEWEB_DOC_CHARS = 1300  # committed documents are at most this long
SWEEP_WIDTH = 32
N_SWEEP_BMP, N_SWEEP_REST = 10, 4  # committed sample of each sweep kind, besides SWEEP_FIXED
MAX_ROW = 16 * 1024  # bytes of a committed row
PACK_MIN = 256  # shorter strings are never written as recipes
SCAN_BEGIN, SCAN_END = 100, 140  # newmm's safe-mode scan window

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


@functools.cache
def environment() -> dict[str, Any]:
    from pythainlp.corpus import corpus_path

    corpus = Path(corpus_path())
    return {
        "datatrove": version("datatrove"),
        "pythainlp": version("pythainlp"),
        "python_crfsuite": version("python-crfsuite"),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "unicode": unicodedata.unidata_version,
        "data_sha256": {
            f"corpus/{name}": sha256_file(corpus / name)
            for name in ("words_th.txt", "sentenceseg_crfcut.model")
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
    from pythainlp.tokenize import crfcut, newmm, tcc_p, word_tokenize

    tok = load_word_tokenizer(LANG)

    def record(text: str) -> dict[str, Any]:
        row: dict[str, Any] = {}
        put(row, "tcc", lambda: [len(x) for x in tcc_p.tcc(text)])
        put(row, "nm", lambda: [len(x) for x in newmm.segment(text)])
        put(row, "nms", lambda: [len(x) for x in newmm.segment(text, safe_mode=True)])
        try:
            toks = word_tokenize(text)
        except Exception as e:
            row["x_toks"] = row["x_lab"] = err(e)
        else:
            row["toks"] = [len(x) for x in toks]
            if toks:
                put(row, "lab", lambda: "".join(crfcut._tagger.tag(crfcut._extract_features(toks))))
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
        stream = self.stream(items, order, 8)
        for done, (i, line) in enumerate(zip(order, stream, strict=True), 1):
            lines[i] = line
            if done % 10_000 == 0:
                log(f"  {what}: {done}/{len(items)} rows, {time.monotonic() - t0:.0f} s")
        log(f"{what}: {len(items)} rows in {time.monotonic() - t0:.1f} s")
        return lines

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


# --------------------------------------------------------------------------
# Thai characters and resources

CONSONANTS = [chr(c) for c in range(0x0E01, 0x0E2F)]  # ก-ฮ
TONES = ["\u0e48", "\u0e49", "\u0e4a", "\u0e4b"]  # ่ ้ ๊ ๋
LEADING = list("เแโใไ")
FOLLOWING = list("ะาำๅ")
ABOVE_BELOW = ["\u0e31", *map(chr, range(0x0E34, 0x0E3B)), "\u0e47", "\u0e4c", "\u0e4d", "\u0e4e"]
THAI_DIGITS = [chr(c) for c in range(0x0E50, 0x0E5A)]
THAI_OTHER = [*"ๆฯ฿๏๚๛", "\u0e00", "\u0e3b", "\u0e5c", "\u0e7f"]

# The alternatives of pythainlp.tokenize.tcc_p._RE_TCC before its placeholders are replaced
# (c: a consonant, t: an optional tone mark, k: an optional final cluster with thanthakhat).
TCC_TEMPLATES = [
    "เc็ck", "เcctาะk", "เccีtยะk", "เccีtย(?=[เ-ไก-ฮ]|$)k", "เcc็ck", "เcิc์ck", "เcิtck",
    "เcีtยะ?k", "เcืtอะ?k", "เc[ิีุู]tย(?=[เ-ไก-ฮ]|$)k", "เctา?ะ?k", "cัtวะk", "c[ัื]tc[ุิะ]?k",
    "c[ิุู]์", "c[ะ-ู]tk", "cรรc์", "c็", "ct[ะาำ]?k", "ck", "แc็c", "แcc์", "แctะ", "แcc็c",
    "แccc์", "โctะ", "[เ-ไ]ct", "ก็", "อึ", "หึ",
]  # fmt: skip
LOOKAHEAD = "(?=[เ-ไก-ฮ]|$)"


def check_tcc_templates() -> None:
    from pythainlp.tokenize import tcc_p

    def expand(tpl: str) -> str:
        k = "(cc?[dิ]?[์])?"
        return (
            tpl.replace("k", k)
            .replace("c", "[ก-ฮ]")
            .replace("t", "[่-๋]?")
            .replace("d", "\u0e39\u0e38")
        )

    if [expand(t) for t in TCC_TEMPLATES] != tcc_p._RE_TCC:
        raise AssertionError("TCC_TEMPLATES no longer match pythainlp.tokenize.tcc_p._RE_TCC")


def class_chars(spec: str) -> list[str]:
    """The characters of a regex class body such as "เ-ไก-ฮ"."""
    out: list[str] = []
    i = 0
    while i < len(spec):
        if i + 2 < len(spec) and spec[i + 1] == "-":
            out += [chr(c) for c in range(ord(spec[i]), ord(spec[i + 2]) + 1)]
            i += 3
        else:
            out.append(spec[i])
            i += 1
    return out


def tcc_instance(tpl: str, rng: random.Random) -> tuple[str, bool]:
    """A random text the TCC alternative `tpl` matches, and whether `tpl` has the lookahead."""
    out: list[str] = []
    look = False
    i = 0
    while i < len(tpl):
        if tpl.startswith(LOOKAHEAD, i):
            look = True
            i += len(LOOKAHEAD)
            continue
        ch = tpl[i]
        if ch == "[":
            j = tpl.index("]", i)
            choices = class_chars(tpl[i + 1 : j])
            i = j + 1
        else:
            i += 1
            if ch == "c":
                choices = CONSONANTS
            elif ch == "t":
                choices = ["", *TONES]
            elif ch == "k":
                choices = ["", *(final_cluster(rng) for _ in range(3))]
            else:
                choices = [ch]
        if i < len(tpl) and tpl[i] == "?":
            choices = ["", *choices]
            i += 1
        out.append(rng.choice(choices))
    return "".join(out), look


def final_cluster(rng: random.Random) -> str:
    c = rng.choice
    return c(CONSONANTS) + c(["", *CONSONANTS]) + c(["", "\u0e39", "\u0e38", "\u0e34"]) + "\u0e4c"


@functools.cache
def dictionary() -> list[str]:
    from pythainlp.corpus import thai_words

    return sorted(thai_words())


def enders_starters() -> tuple[list[str], list[str]]:
    from pythainlp.tokenize import crfcut

    return sorted(crfcut._ENDERS), sorted(crfcut._STARTERS)


# --------------------------------------------------------------------------
# Inputs: handwritten and generated cases

WHITESPACE = [
    " ", "\t", "\n", "\r\n", "\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x1f", "\x85",
    "\u00a0", "\u1680", "\u2000", "\u2001", "\u2002", "\u2003", "\u2004", "\u2005", "\u2006",
    "\u2007", "\u2008", "\u2009", "\u200a", "\u2028", "\u2029", "\u202f", "\u205f", "\u3000",
]  # fmt: skip
# Not whitespace for str.isspace() and the regex's \s, although they look like it.
NOT_SPACE = ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u180e", "\xad"]

AMBIGUOUS = [
    "ตากลม", "ตากลมตากลม", "คนขับรถ", "ไปหามเหสี", "การบินไทย", "มากกว่า", "นายกรัฐมนตรี",
    "ทดสอบการตัดคำ", "สวัสดีครับ", "ฉันรักภาษาไทย", "ประเทศไทย", "หมอกควัน", "กินข้าวกับปลา",
    "ยาลดไข้", "ตาก", "กลม", "ตา", "มารดา", "กรุงเทพมหานคร", "ข้าวมันไก่", "มันฝรั่งทอด",
    "แม่ค้าขายผลไม้", "รถไฟฟ้าใต้ดิน", "โรงเรียนอนุบาล", "นักเรียนนักศึกษา", "ผู้ใหญ่บ้าน",
    "คณะกรรมการการเลือกตั้ง", "ตากลมกลมตา", "กลมกลืน", "ไหมไทย", "ไหม", "เสือนอนกิน",
    "ขอบคุณมากครับ", "วันนี้อากาศดีมาก", "ผมชื่อสมชาย", "โอเคบ่พวกเรารักภาษาบ้านเกิด",
    "ชินโซ อาเบะ เกิด 21 กันยายน", "วรรณกรรม ภาพวาด และการแสดงงิ้ว ",
    "เงิน1,234บาท19:32น 127.0.0.1", "ขณะนี้เวลา 12:00น อัตราแลกเปลี่ยน 1,234.5 baht/zeny",
    "IP address ของคุณคือ 127.0.0.1 ครับ", "พ.ศ. 2567", "ค.ศ. 2024", "ดร.สมชาย", "น.ส.สมหญิง",
    "กทม.", "ก.ค.", "ฯลฯ", "กรุงเทพฯ", "ไปๆมาๆ", "ไป ๆ มา ๆ", "ก้ม ๆ เงย ๆ", "ก ข ไม่กระดิกหู",
]  # fmt: skip

OOV_CASES = [
    "ฮฮฮฮ", "ฆฆฆ", "ฬฬ", "ฃฃฃ", "ฅฅ", "ฦๅ", "ฤๅ", "ฮฮฮบ้าน", "ฮฮฮกบ", "ฮฮฮabc", "ฮฮฮ123", "ฮฮฮ ก",
    "ฮฮฮ", "ฮฮฮ\n", "ฮฮฮ\rก", "ฮฮฮ\tก", "ฮฮฮé", "ฮฮฮ😀", "ฮฮฮ-", "ฮฮฮ๑๒", "ฮฮฮไป", "ฮฮฮกิน",
    "ฆฆฆบ้านฆฆฆ", "ฮฮฮกกฮฮฮ", "ฮฮฮนาฮฮฮ", "ฮฮฮฯ", "ฮฮฮๆ", "ฬฬฬ.", "ฮ่ฮ่ฮ่", "ฮ็ฮ็",
    "เฮฮฮ", "ฮฮฮา", "ฮฮฮะ", "ฮฮฮ์", "ฮฮฮบ", "ฮฮฮกข", "ฮฮฮคน", "ฮฮฮกขค",
]  # fmt: skip

NONTHAI_CASES = [
    "abc", "a-b", "-", "--", "-a-", "a1", "1a", "12,345.67", "1,2,3", "1.", ".1", "1..2", "1,",
    ",1", "๑๒๓", "๑,๒๓๔.๕", "１２３", "²", "٣٤", " ", "  ", "\t", " \t ", "\n", "\r\n", "\r",
    "\n\n", "\r\r\n", "\x0b", "\x0c", "\xa0", "\u200b", "\u3000", "é", "日本語", "😀", "@#$",
    "a b", "abc def", "abcไทย", "ไทยabc", "ไทย abc ไทย", "ไทย123ไทย", "ไทย\tไทย", "ไทย\r\nไทย",
    "ไทย\rไทย", "\rไทย", "ไทย\r", "ไทย\n", "\nไทย", "ไทย\n\nไทย", "ไทย  ไทย", "ไทย \t ไทย",
    "ไทย\xa0ไทย", "ไทย\u200bไทย", "ไทย\u3000ไทย", "ไทย-ไทย", "a-ไทย", "ไทย-a", "ไทย.ไทย",
    "ไทย,ไทย", "A-Z", "x-ray", "e-mail", "COVID-19", "COVID19", "19COVID", "ไทย日本", "日本ไทย",
    "ไทยÄ", "Äไทย", "ไทย'ไทย", '"ไทย"', "(ไทย)", "[ไทย]", "ไทย!", "ไทย?", "ไทย...", "...ไทย",
    "ไทย:ไทย", "ไทย;ไทย", "aั", "1่", "-็", " ์", "\nั", "ไทย\x0bไทย",
]  # fmt: skip

NUM_CASES = [
    "12:30", "1,234.5", "127.0.0.1", "1.2.", ":5", "5:", "1:2:3", "x1:2", "12.30.45", "๑๒:๓๐",
    "1 : 2", "1:2 3:4", "1:2abc", "abc1:2", "12,", "12,a", "1.2.3.4.5.6", "3.14159", "0.5%",
    "1,000,000บาท", "2,5", "1:", "a:1", "1:a", "ก1:2", "1:2ก", "1:2:", "1::2", "1.:2",
    "1,2.3:4", "เวลา 12:30 น.", "1,234.5บาท", "ราคา1,234บาท", "เวลา12:30น.", "ip127.0.0.1x",
    "๑.๒.๓", "1.๒", "๑:2", "１２:３０", "12:３０", "1²:2", "1:²", "12:30:45.67", "1,2,3,4,5",
    "1.2.3.", "ไทย1.2ไทย", "ไทย1:2ไทย", "ไทย 1 , 2 ไทย", "1,,2", "1..2", "1.2..3", "10:00-12:00",
    "1-2:3", "-1.5", "+1.5", "1e5", "1.5e10", "12:30น.", "x:1:2:y", "1:2:3:4:5:6:7:8",
]  # fmt: skip

SENT_CASES = [
    "ไปไหน?", "ไป!", "ไป.", "ไป...", "ไป?!", "ไป ?", "?", ".", "!", " . ", "ไป.ไป", "ไป. ไป",
    "abc. def", "1.2. 3", "ไป\x0b.", "ไป!\n\nไป", " ไป", "  ไป", "ไป.  ไป", "ไป. \nไป", "\nไป",
    "ไป ", "ไป.  ", "ไปครับ \n", "   ", "\n", "\t", "ไป\n", "ไป \n ", ". ไป", "! ไป", " ! ไป",
    "ไป!!", "ไป!?", "?!ไป", "ไป.)", "ไป.\"", "ไป ? ", "ไป\t?\t", "ไปไหน? ไปตลาด! แล้วไง.",
    "ผมไปตลาดครับ คุณไปไหนคะ", "วันนี้อากาศดีมาก พรุ่งนี้ฝนตก", "ฉันไปโรงเรียน และเขาไปทำงาน",
    "ผมกินข้าวแล้ว คุณกินหรือยัง", "เขามาแล้ว เราไปกันเถอะ", "ไม่ใช่ แต่ว่าใช่",
    "สวัสดีครับ ผมชื่อสมชาย ยินดีที่ได้รู้จักครับ", "ทำไม ทำไม ทำไม", "นั้น นี้ นั้น นี้",
]  # fmt: skip

TOKEN_CASES = [
    "ไป\x00มา", "\x00", "\x00\x00", "a\x00b", "ไป \x00 มา", "a|b", "ไป|มา", "x=y", "ไป=มา", "=",
    "|", "||", "a=b|c", "word_1_0_1=x", "ender", "starter", "normal", "xxpad", "bias",
    "ไปxxpadมา", "xxpad xxpad", "ender|normal", "ไป:มา", "a:b", "ไป\\มา", "ไป\x7fมา", "ไป\x1bมา",
    "😀", "ไป😀มา", "👍🏽", "🇹🇭", "👨\u200d👩\u200d👧", "❤️", "1️⃣", "ก́",
    "\u0e31", "\u0e48", "\u0e48\u0e48", "ก่่", "ก่้", "กำ", "กํา", "เเก",
    "ก\u200bข", "ก\ufeffข", "ก\xadข",
    "ๆ", "ไปๆ", "ๆไป", "ฯ", "฿100", "๏", "ไป๚", "ไป๛", "ก\u0e3b", "\u0e5cก", "ก\u0e7f", "ไทย�",
]  # fmt: skip


# Runs that stay ambiguous long enough for newmm's graph to pass 50 edges (its cutoff).
CUTOFF_CASES = [
    *("ส" * n for n in (20, 30, 40, 50, 51, 52, 60, 100)),
    *("นา" * n for n in (20, 25, 26, 30, 40)),
    *("ลา" * n for n in (25, 40, 80)),
    "ไป" + "ส" * 60 + "มา", "ส" * 60 + " " + "นา" * 30, "นา" * 30 + "ส" * 60, "ส" * 60 + "abc",
    "ส" * 60 + "\n" + "ส" * 60, "ครับ" + "นา" * 40 + "ครับ", "ส" * 45 + "นา" * 20 + "ลา" * 20,
]  # fmt: skip


def safe_items(rng: random.Random, docs: Sequence[str]) -> list[str]:
    """Texts around newmm's safe-mode cut: lengths and spaces near the 100..140 window."""
    nows = re.sub(r"\s+", "", "".join(docs[:20]))
    out: list[str] = []
    for n in (139, 140, 141, 280, 281):
        at = rng.randrange(len(nows) - n)
        out.append(nows[at : at + n])
    base = nows[:170]
    out += [base[:pos] + " " + base[pos:] for pos in (98, 99, 100, 101, 120, 138, 139, 140, 141)]
    out += [
        base[:pos] + " " + base[pos : pos + 20] + " " + base[pos + 20 :] for pos in (100, 120, 139)
    ]
    out += [
        base[:115] + sep + base[115:] for sep in ("\t", "\xa0", "\n", "\u3000", "  ", "-", "a", "1")
    ]
    out.append(base[:100] + "abc def" + base[100:])
    out.append(base[:100] + "1,234.5" + base[100:])
    out.append(base[:99] + "ฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮฮ" + base[99:])
    out.append("ก" * 100 + "".join(rng.choice(dictionary()) for _ in range(12)))
    out.append("ตากลม" * 60)
    out.append("a" * 139 + "ไทย")
    out.append(" " * 141)
    out.append("ไทย " * 50)
    return out


def tcc_items(rng: random.Random) -> list[str]:
    out: list[str] = []
    for tpl in TCC_TEMPLATES:
        for _ in range(2):
            inst, look = tcc_instance(tpl, rng)
            out += [inst, inst + "ก", "ก" + inst, inst + "\n", inst + "\nก", inst * 2]
            out.append(inst + " " + inst)
            if look:
                out += [inst + "เ", inst + "า", inst + "\n\n", inst + "\r\n", inst + "ไป"]
    for b in range(0x0E00, 0x0E80, 16):
        block = [chr(c) for c in range(b, b + 16)]
        out += [" ".join(f"ก{c}" for c in block), " ".join(f"เก{c}า" for c in block)]
    return out


def case_items(fineweb_texts: Sequence[str]) -> list[Item]:
    check_tcc_templates()
    rng = random.Random(f"{SEED}/cases")
    words = dictionary()
    enders, starters = enders_starters()
    out: list[Item] = []

    def add(kind: str, *texts: str) -> None:
        out.extend(({"k": kind}, t) for t in texts)

    add("empty", "")
    add("tcc", *tcc_items(rng))
    for w in WHITESPACE + NOT_SPACE:
        add("ws", w, w * 3, f"ไทย{w}ไทย", f"{w}ไทย", f"ไทย{w}", f"ไป.{w}มา", f"1:{w}2", f"{w}.{w}")
    add("ws", "".join(WHITESPACE), " \t\n\r\x0b\x0c", " \xa0 ", "\r\n\r\n", " \n ", "\n \n")
    add("dict", *rng.sample(words, 50))
    add("dict", *("".join(rng.sample(words, rng.randint(2, 6))) for _ in range(30)))
    add("dict", *(" ".join(rng.sample(words, rng.randint(2, 6))) for _ in range(16)))
    odd = [w for w in words if re.search(r"[^\u0e00-\u0e7f]", w)]
    for e in rng.sample(odd, 24):
        add("dict-odd", e, f"ไป{e}มา", f"{e} {e}")
    add("ambiguous", *AMBIGUOUS)
    add("oov", *OOV_CASES)
    add("nonthai", *NONTHAI_CASES)
    add("num", *NUM_CASES)
    add("sent", *SENT_CASES)
    for e in enders:
        add("ender", f"ไปตลาด{e} ผมกลับบ้าน", f"ไป{e}.")
    for s in starters:
        add("starter", s, f"ผมกลับบ้านแล้ว {s}ไปตลาด")
    add("token", *TOKEN_CASES)
    add("safe", *safe_items(rng, fineweb_texts))
    runs = ["ก", "กา", "ตากลม", "ไทย ", "ฮ", "ๆ", "a", "1", "1,", " ", "\n", "-", "😀", "ก่", "เ"]
    runs += ["ครับ", "ครับ ", "ไป.", "\x00", "|", "\u200b", "ก\u200b"]
    for r in runs:
        add("long", r * max(1, 100 // len(r)))
    add("cutoff", *CUTOFF_CASES)

    # FineWeb-2 lines in odd contexts.
    lines = [x for t in fineweb_texts[:40] for x in t.splitlines() if 8 <= len(x) <= 120]
    for line in rng.sample(lines, min(8, len(lines))):
        add("fw-ctx", f"  {line}  ", line.replace(" ", ""), line.replace(" ", "\xa0"))
        add("fw-ctx", line + "!", line.replace(" ", "\n"), "\u200b".join(line))
    return out


# --------------------------------------------------------------------------
# Inputs: FineWeb-2, fuzz, sweep, long texts


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
    return [({"d": d}, doc) for d, doc in enumerate(docs)]


def sample_docs(items: Sequence[Item]) -> list[Item]:
    rng = random.Random(f"{SEED}/fineweb-docs")
    fits = [i for i, (_, text) in enumerate(items) if len(text) <= FINEWEB_DOC_CHARS]
    return [items[i] for i in sorted(rng.sample(fits, min(N_FINEWEB_DOCS, len(fits))))]


# Blocks the committed sweep always has: controls and whitespace, ASCII, Latin-1, the Thai
# block, general punctuation, variation selectors, emoji.
SWEEP_FIXED = [
    0x0, 0x20, 0x40, 0x60, 0x80, 0xA0, 0xE00, 0xE20, 0xE40, 0xE60, 0x2000, 0x3000, 0xFE00,
    0xFF00, 0x1F600,
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
            "\n".join(f"ก{chr(c)}ขา {chr(c)}ไทย{chr(c)}" for c in range(b, b + SWEEP_WIDTH)),
        )
        for b in kinds["ctx"]
    ]
    return items


def fuzz_items(fineweb_texts: Sequence[str]) -> list[Item]:
    rng = random.Random(f"{SEED}/fuzz")
    enders, starters = enders_starters()
    words = dictionary()
    pools: dict[str, list[str]] = {
        "cons": CONSONANTS,
        "lead": LEADING,
        "follow": FOLLOWING,
        "mark": ABOVE_BELOW,
        "tone": TONES,
        "tdigit": THAI_DIGITS,
        "tother": THAI_OTHER,
        "digit": list("0123456789"),
        "sep": list(",.:"),
        "latin": [chr(c) for c in range(0x41, 0x7B) if chr(c).isalpha()] + ["-"],
        "ws": WHITESPACE + NOT_SPACE,
        "punct": list("!?.|=;()\"'[]{}<>/\\#@%&*+_~^`"),
        "nul": ["\x00", "\x01", "\x7f"],
        "emoji": ["😀", "👍", "🇹🇭", "👨\u200d👩\u200d👧", "❤️", "1️⃣"],
        "comb": ["\u0301", "\u0308", "\u200d", "\ufe0f", "🏽"],
        "other": list("é日本ЖαÄ½²①"),
        "ender": enders,
        "starter": starters,
    }
    names = [*pools, "word", "word", "word", "word", "fineweb", "fineweb", "tcc"]
    items: list[Item] = []
    while len(items) < N_FUZZ:
        pieces = []
        for _ in range(rng.randint(1, 8)):
            name = rng.choice(names)
            if name == "fineweb":
                doc = rng.choice(fineweb_texts)
                at = rng.randrange(len(doc))
                piece = doc[at : at + rng.randint(1, 16)]
            elif name == "word":
                piece = rng.choice(words)
            elif name == "tcc":
                piece = tcc_instance(rng.choice(TCC_TEMPLATES), rng)[0]
            else:
                piece = "".join(rng.choice(pools[name]) for _ in range(rng.randint(1, 3)))
            if rng.random() < 0.06:
                piece *= rng.randint(2, 4)
            pieces.append(piece)
            if rng.random() < 0.3:
                pieces.append(rng.choice([" ", " ", "\n", ". ", "!", "?"]))
        items.append(({}, "".join(pieces)[: rng.choice([8, 20, 32])]))
    return items


def long_items(docs: Sequence[str]) -> list[Item]:
    """Long texts: safe mode's chunking and CRFCut over many tokens."""
    rng = random.Random(f"{SEED}/long")
    words = dictionary()
    items: list[Item] = []

    def add(kind: str, text: str) -> None:
        items.append(({"k": kind}, text))

    def joined(sep: str, limit: int, start: int = 0) -> str:
        out: list[str] = []
        size = 0
        for doc in docs[start:]:
            if size >= limit:
                break
            out.append(doc)
            size += len(doc) + len(sep)
        return sep.join(out)[: max(limit, 1)]

    short = [d for d in docs if 140 <= len(d) <= 650]
    for i in range(0, 8, 2):
        add("FineWeb-2 documents, two short ones joined", short[i] + "\n" + short[i + 1])
        add("FineWeb-2 documents, two short ones, no separator", short[i] + short[i + 1])
    nows = re.sub(r"\s+", "", "".join(docs[:60]))
    for n in (150, 300, 600, 900, 1200):
        at = rng.randrange(len(nows) - n)
        add(f"FineWeb-2 text without whitespace, {n} characters", nows[at : at + n])
    for n in (300, 600, 1000):
        add(
            f"dictionary words without spaces, {n} characters",
            "".join(rng.choice(words) for _ in range(n))[:n],
        )
        add(
            f"dictionary words with spaces, {n} characters",
            " ".join(rng.choice(words) for _ in range(n))[:n],
        )
    for sep, name in (("\n", "newlines"), (" ", "spaces"), ("", "nothing")):
        for limit in (3000, 12_000, 60_000):
            add(f"FineWeb-2 documents joined by {name}, {limit} characters", joined(sep, limit))
    for n in (5000, 20_000, 60_000):
        add(f"FineWeb-2 text without whitespace, {n} characters", nows[:n])
    add(
        "dictionary words without spaces, 20000 characters",
        "".join(rng.choice(words) for _ in range(20_000))[:20_000],
    )
    add(
        "dictionary words with spaces, 20000 characters",
        " ".join(rng.choice(words) for _ in range(20_000))[:20_000],
    )
    sentence = "ผมไปตลาดเมื่อวานนี้ครับ แล้วก็ซื้อผลไม้มาด้วย! คุณจะไปไหม? "
    add("sentences", sentence * (20_000 // len(sentence)))
    for r in ("ตากลม", "ก", "กา", "ฮ", "ๆ", "ไทย ", "a", "1,", " ", "\n", "ครับ "):
        add(f"run of {r!r}", r * (6000 // len(r)))
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
    emit("long", long_items(docs), source=src)
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
    s = sub.add_parser("record", help="cases, FineWeb-2 documents, fuzz, sweep and long texts")
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
