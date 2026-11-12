"""Record what SudachiPy 0.7.0 and datatrove's Japanese word tokenizer return.

The goldens record the output of SudachiPy 0.7.0 (Apache-2.0), © Works Applications Co.,
Ltd., with SudachiDict-core 20261015 (Apache-2.0), © Works Applications Co., Ltd., and of
datatrove 0.10.1 (Apache-2.0), © HuggingFace. The FineWeb-2 excerpts are from
HuggingFaceFW/fineweb-2 (ODC-By 1.0), subset jpn_Jpan.

`pyroid_core::sudachi` must tokenize exactly as SudachiPy 0.7.0 does (split mode A, the
default configuration of `sudachipy.Dictionary()`), and pyroid's Japanese word tokenizer
exactly as datatrove's `load_word_tokenizer("jpn_Jpan")` (spaCy's `ja` pipeline with
datatrove's `datatrove.ja.JapaneseTokenizer` and the sentencizer). This script runs both on
fixed inputs and writes what they return next to it; the replay tests feed every row to the
Rust code and require identical results.

Run inside the pyroid-dev container (never on the host) with an interpreter that has
sudachipy==0.7.0, sudachidict-core==20261015, datatrove==0.10.1, spacy==3.8.18 and pyarrow:

    python oracle.py tags [--out DIR]
    python oracle.py chunks [--out DIR] [--reverse]
    python oracle.py morphemes [--out DIR] [--fineweb PARQUET] [--full] [--reverse]
    python oracle.py datatrove [--out DIR] [--fineweb PARQUET] [--full] [--reverse]
    python oracle.py large PARQUET --out DIR [--reverse]
    python oracle.py compare DIR1 DIR2

Without `--out` the files are written next to this script; `--full` (which needs `--out`)
records every row instead of the committed selection. `--reverse` processes the rows in
reverse order and writes them in the usual order: the output must not change (`compare`
checks that two directories hold identical files). `large` records every line (morphemes)
and every document (datatrove) of a FineWeb-2 parquet file. Every run gives the same bytes.

Files, one text per row:

    morphemes-cases.jsonl     handwritten and generated inputs for every special case of the
                              Sudachi pipeline (rewrite.def, NFKC and lowercasing, prolonged
                              sound marks, yomigana, numerics, katakana OOV, char.def
                              categories, whitespace, emoji, controls, input limits, ...)
    morphemes-fineweb.jsonl   FineWeb-2 lines: `str.splitlines(keepends=True)` of each
                              document, lines over 40000 bytes cut at character boundaries
    morphemes-fuzz.jsonl      seeded random strings
    morphemes-sweep.jsonl     code point sweep: every 32 consecutive code points ("run"), and
                              each code point between two kanji ("ctx": "日{c}本" for 32 code
                              points, joined by "\\n"); surrogates left out
    datatrove-cases.jsonl     the same inputs as morphemes-cases.jsonl
    datatrove-fineweb.jsonl   whole FineWeb-2 documents
    datatrove-fuzz.jsonl      the same inputs as morphemes-fuzz.jsonl
    datatrove-sweep.jsonl     the same inputs as morphemes-sweep.jsonl
    datatrove-long.jsonl      documents over 40000 bytes (datatrove tokenizes them in chunks),
                              many with a chunk boundary placed at a chosen character
    chunks.jsonl              datatrove's chunk_text_on_bytes on random texts
    tags.json                 every POS of the dictionary and its spaCy tag

The committed files hold a selection: rows of at most 16 KiB, a seeded sample of the FineWeb-2
lines and documents and of the sweep. `--full` writes every row (the whole sweep is about
70,000 rows per level).

Every JSONL file starts with a header line `{"header": {...}}`: the level, the generator and
seed, the selection, the versions of SudachiPy, SudachiDict-core, datatrove, spaCy, Python
and its Unicode database, the sha256 of the system.dic used and of SudachiPy's resource files,
the source parquet file (if any) and the number of rows. Then one row per line:

    morphemes:  {"t": text, "m": [[begin, end, pos_id, word_id], ...]}
    datatrove:  {"t": text, "w": word_tokenize(t), "sp": span_tokenize(t),
                 "s": sent_tokenize(t)}
    chunks:     {"t": text, "n": max_chunk_size, "c": [len(chunk), ...]}

In "m", begin and end are `Morpheme.begin()` and `Morpheme.end()` (character offsets in the
original text), pos_id is `Morpheme.part_of_speech_id()` and word_id `Morpheme.word_id()`, in
the order `Tokenizer.tokenize` returns them, empty-surface morphemes included. "sp" holds the
spans as [start, end] pairs. A datatrove row of a text over 40000 bytes also has
"c": [len(chunk) for chunk in chunk_text_on_bytes(t, 40000)]. When the call raises, the row
has "x": "<exception type>: <message>" instead of the results (a datatrove row keeps "c").
Rows can also have "k" (what the input is), "d" (the FineWeb-2 row), "l" (the line in that
document), "p" (the piece of a line cut at 40000 bytes) or "cp" (the first code point of a
sweep row).

A text ("t", and each string in "w" and "s") is a JSON string or, when that is much shorter,
a recipe: an array of items that are strings or [string, count] pairs; the text is the
concatenation of the strings, each pair's string repeated count times. For example
[["ア", 3], " x"] is "アアア x".

`tags.json` holds the header, then "pos": [[pos_id, [six POS fields], tag, in_tag_map], ...]
for every POS id of the dictionary, where tag is
`"-".join([x for x in pos[:4] if x != "*"])` (what datatrove's tokenizer sets as the token
tag) and in_tag_map tells whether `spacy.lang.ja.tag_map.TAG_MAP` has it (its `resolve_pos`
raises KeyError for tags outside it), and "missing": the POS ids whose tag it lacks.

Texts containing "IS_ALPHA" are left out everywhere: spaCy's results on them can depend on
what it tokenized before (datatrove retries a failing call without that string).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
import platform
import random
import re
import sys
import time
import unicodedata
import warnings
from collections import Counter
from collections.abc import Callable, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

# Look-alike and invisible characters on purpose.
# ruff: noqa: RUF001

HERE = Path(__file__).resolve().parent
SEED = 20261109
FINEWEB = Path("/data/fineweb2/jpn_Jpan.parquet")
FINEWEB_DATASET = "HuggingFaceFW/fineweb-2 jpn_Jpan (ODC-By 1.0)"
DT_CHUNK = 40_000  # datatrove: chunk_text_on_bytes(text, 40000) for Japanese
MAX_INPUT = 49_149  # sudachi.rs: at most u16::MAX / 4 * 3 bytes of input
MAX_EDITED = 65_535  # sudachi.rs: at most u16::MAX bytes after each input text plugin
N_FUZZ = 2000
N_FINEWEB_LINES = 450  # committed sample of FineWeb-2 lines
N_FINEWEB_DOCS = 30  # committed sample of FineWeb-2 documents
SWEEP_WIDTH = 32
N_SWEEP_BMP, N_SWEEP_REST = 24, 8  # committed sample of each sweep kind
MAX_ROW = 16 * 1024  # bytes of a committed row
PACK_MIN = 256  # shorter strings are never written as recipes
SKIP = "IS_ALPHA"
CHUNK_PLAN = {10: 250, 1000: 250, 10_000: 150, 10_001: 100, 40_000: 1200, 1_000_000: 40}

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
    import sudachipy

    # What sudachipy.Dictionary() loads with the default configuration.
    dic = Path(sudachipy._find_dict_path("core"))
    resources = Path(sudachipy._DEFAULT_RESOURCEDIR)
    return {
        "sudachipy": version("sudachipy"),
        "sudachidict_core": version("sudachidict-core"),
        "system_dic_sha256": sha256_file(dic),
        "resources_sha256": {
            p.name: sha256_file(p) for p in sorted(resources.iterdir()) if p.is_file()
        },
        "datatrove": version("datatrove"),
        "spacy": version("spacy"),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "unicode": unicodedata.unidata_version,
        "split_mode": "A",
    }


def header(level: str, generator: str, selection: str, **extra: Any) -> dict[str, Any]:
    return {
        "level": level,
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
    kept = [line for line in lines if len(line.encode("utf-8")) <= MAX_ROW]
    if len(kept) < len(lines):
        log(f"  {len(lines) - len(kept)} rows over {MAX_ROW} bytes left out")
    return kept


# --------------------------------------------------------------------------
# Recording


def morpheme_recorder() -> Recorder:
    import sudachipy

    tok = sudachipy.Dictionary().tokenizer(mode=sudachipy.SplitMode.A)

    def record(text: str) -> dict[str, Any]:
        try:
            ms = tok.tokenize(text)
        except Exception as e:
            return {"x": err(e)}
        return {"m": [[m.begin(), m.end(), m.part_of_speech_id(), m.word_id()] for m in ms]}

    return record


def datatrove_recorder() -> Recorder:
    from datatrove.utils.word_tokenizers import chunk_text_on_bytes, load_word_tokenizer

    tok = load_word_tokenizer("jpn_Jpan")

    def record(text: str) -> dict[str, Any]:
        row: dict[str, Any] = {}
        if utf8len(text) > DT_CHUNK:
            row["c"] = [len(c) for c in chunk_text_on_bytes(text, DT_CHUNK)]
        try:
            words = tok.word_tokenize(text)
            spans = tok.span_tokenize(text)
            sents = tok.sent_tokenize(text)
        except Exception as e:
            row["x"] = err(e)
            return row
        row["w"] = [pack(w) for w in words]
        row["sp"] = [list(s) for s in spans]
        row["s"] = [pack(s) for s in sents]
        return row

    return record


def clean(items: Sequence[Item]) -> list[Item]:
    kept = [it for it in items if SKIP not in it[1]]
    if len(kept) < len(items):
        log(f"  {len(items) - len(kept)} texts containing {SKIP!r} left out")
    return kept


def record_items(items: Sequence[Item], record: Recorder, reverse: bool, what: str) -> list[str]:
    lines = [""] * len(items)
    order = range(len(items) - 1, -1, -1) if reverse else range(len(items))
    t0 = time.monotonic()
    for done, i in enumerate(order, 1):
        meta, text = items[i]
        lines[i] = dumps({**meta, "t": pack(text), **record(text)})
        if done % 10_000 == 0:
            log(f"  {what}: {done}/{len(items)} rows, {time.monotonic() - t0:.0f} s")
    log(f"{what}: {len(items)} rows in {time.monotonic() - t0:.1f} s")
    return lines


# --------------------------------------------------------------------------
# Sudachi resources


def resource(name: str) -> str:
    import sudachipy

    return (Path(sudachipy._DEFAULT_RESOURCEDIR) / name).read_text(encoding="utf-8")


def rewrite_def() -> tuple[list[str], list[str]]:
    """The ignore-normalize characters and the replace keys of rewrite.def, in file order."""
    ignore: list[str] = []
    replace: list[str] = []
    for raw in resource("rewrite.def").split("\n"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split()
        (ignore if len(cols) == 1 else replace).append(cols[0])
    return ignore, replace


def char_def() -> list[tuple[int, int, frozenset[str]]]:
    """[begin, end) segments of char.def with their categories (unions, as sudachi.rs does)."""
    ranges = []
    for raw in resource("char.def").split("\n"):
        line = raw.strip()
        if not line.startswith("0x"):
            continue
        cols = line.split()
        r = cols[0].split("..")
        begin = int(r[0], 16)
        end = int(r[1], 16) + 1 if len(r) > 1 else begin + 1
        cats = list(itertools.takewhile(lambda c: not c.startswith("#"), cols[1:]))
        ranges.append((begin, end, frozenset(cats)))
    bounds = sorted({0, 0x110000} | {b for b, _, _ in ranges} | {e for _, e, _ in ranges})
    segs = []
    for lo, hi in itertools.pairwise(bounds):
        cats = frozenset().union(*(c for b, e, c in ranges if b <= lo and hi <= e))
        segs.append((lo, hi, cats or frozenset({"DEFAULT"})))
    return segs


def stable(ch: str) -> bool:
    """Whether Sudachi's input normalization leaves `ch` alone."""
    return (
        unicodedata.category(ch) not in ("Cn", "Cs", "Co")
        and unicodedata.normalize("NFKC", ch) == ch
        and ch.lower() == ch
    )


def category_representatives() -> list[tuple[str, frozenset[str]]]:
    """One stable character per distinct char.def category set, in code point order."""
    reps: dict[frozenset[str], str] = {}
    for lo, hi, cats in char_def():
        if cats in reps:
            continue
        for cp in range(lo, min(hi, lo + 4096)):
            if stable(chr(cp)):
                reps[cats] = chr(cp)
                break
    return sorted(((ch, cats) for cats, ch in reps.items()), key=lambda x: x[0])


def meets(a: frozenset[str], b: frozenset[str]) -> bool:
    return bool(a & b) or "ALL" in a or "ALL" in b


# --------------------------------------------------------------------------
# Inputs: handwritten and generated cases

WHITESPACE = [
    " ", "\u3000", "\t", "\n", "\r\n", "\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x1f",
    "\x85", "\u2028", "\u2029", "\u00a0", "\u200b", "\u1680", "\u2000", "\u2007", "\u2009",
    "\u202f", "\u205f", "\u180e", "\u2060", "\ufeff",
]  # fmt: skip

NFKC_CASES = [
    "ＡＢＣ", "ａｂｃ", "Ｔｏｋｙｏ２０２０", "ﾃｽﾄ", "ｶﾞｷﾞｸﾞｹﾞｺﾞ", "ﾊﾟﾋﾟﾌﾟﾍﾟﾎﾟ", "ｳﾞｧｲｵﾘﾝ",
    "ﾃﾞｰﾀ", "ｶ\u3099", "か\u3099", "ｶ゛", "ｶﾟﾞ", "ﬁｎａｌ", "ﬀ ﬂ ﬃ ﬄ ﬅ ﬆ", "㍿",
    "株式会社㍿", "①②③⑳㉑㊿", "⑴⒜⒈", "ⅠⅡⅢⅣⅫ", "ⅰⅱⅲ", "ⅯⅭⅮ", "½¼¾⅓⅞", "x²+y³", "H₂O", "℃℉",
    "№", "℡", "™", "㌀㌔㍉", "㋿", "㍻", "㈱", "㊤", "Å", "Ω", "K", "\u00b5", "\u017f", "ẛ",
    "ǅ", "ǈ", "ǋ", "ŉ", "İstanbul", "İ", "ı", "ΣΑΣ", "ΟΔΥΣΣΕΥΣ", "Σ", "σς", "ẞ", "ß", "𝐀𝐁𝐂", "𝒜",
    "𝔸", "Ⓐⓑ", "🄰", "🅰", "¨", "¯", "´", "˜", "΄", "᾽", "ﾞ",
    "ﾟ", "゛゜", "\u3099\u309a", "ｰ", "･", "｡｢｣､", "“”‘’", "＂＇", "＼", "￥", "￦", "￡￠",
    "\u2002\u2003\u2009", "ｱ\u00a0ｲ", "․‥…", "．．", "Ǆǅǆ", "ﬓ", "ﭐ", "ﷺ",
    "ﷻ", "﷼", "שׁ", "\u2126\u212a\u212b", "Ꜳ", "Ꭰ", "ꭰ", "Ⴀ", "Ა", "𐐀", "𞤀", "ᲀ",
    "ǰ", "ΐ", "ꟈ", "Ɐ", "Ʞ", "Ɥ", "ǈǋ", "Ⅼⅼ", "ꞵ", "Ꞵ", "\ua7cb", "\ua7dc", "\U00010570",
    "Ｈｅｌｌｏ\u3000Ｗｏｒｌｄ", "ＩＳ", "ＴＥＳＴテスト", "ｶﾀｶﾅとカタカナ",
]  # fmt: skip

CASE_CASES = [
    "ABC", "Tokyo", "東京TOWER", "ÀÉÎÕÜ", "ĀĂĄ", "ΑΒΓ", "АБВ", "ǄǇǊ", "ΑΣ ΣΑ ΣΑΣ.",
    "Ɜ", "MiXeD CaSe テキスト", "ǅungla", "ÆØÅ", "ĲŒ", "ẞTRASSE", "ŸÿÞþ", "ΆΈΉΊΌΎΏ", "ЀЁЂЃ",
    "ӀӁ", "ԀԂ", "ⰀⰁ", "ⲀⲂ", "ꙀꙂ", "ꚀꚂ", "ＡＢＣＤ", "Ｚ", "𝐙", "ﬀＡ", "ÅÅ", "KK",
]  # fmt: skip

PSM_CASES = [
    "ー", "ーー", "ーーー", "ーーーー", "ーーーーー", "-", "--", "---", "----", "⁓⁓", "〜〜",
    "〰〰", "ー-〜⁓〰", "ラーメン", "ラーーメン", "ラーーーーメン", "すごーーい", "すご〜〜い",
    "すご~~い", "すご～～い", "a--b", "1--2", "--a", "a--", "ｰｰ", "ｰｰｰ", "－－", "‐‐", "‑‑",
    "–", "——", "−−", "ー ー", "ーa ー", "カー", "カーー", "ーカ", "ーーカ", "ｶｰｰ", "ﾗｰﾒﾝ",
    "〜", "～", "~", "- - -", "-ー-", "ーｰ-－〜", "長ーーーい", "ああーーーっ", "ーーーー漢字",
    "漢字ーーーー", "1-1", "1--1", "東京--大阪", "コーーヒー", "ｺｰｰﾋｰ", "ー\u0301ー",
]  # fmt: skip

NUMERIC_CASES = [
    "0", "00", "007", "〇", "〇〇七", "一", "十", "十一", "二十", "百", "千", "万", "一万",
    "十万", "百万", "千万", "一億", "一兆", "万万", "億万", "兆兆", "十二万三千", "二千二十四",
    "二〇二四", "二〇二四年", "一二三", "三十五万六千七百八十九",
    "一億二千三百四十五万六千七百八十九", "九千九百九十九兆", "千兆", "一京", "1万", "1万2000",
    "1億2千万", "3千", "10万", "1.5万", "1,000", "1,000,000", "1,000,000.5", "1,00", "1,0000",
    "12,34", ",123", "1,", "1,,000", "0,123", "000,000", "1.", ".5", "1.2.3", "1..2",
    "3.14159", "0.5", "00.5", "1,000.5", "1,000.000,5", "1.000,5",
    "12,345,678,901,234,567,890", "１２，３４５", "１，０００", "１．５", "１２３",
    "１２３４５６７８９０", "１2３", "十二", "12月25日", "2024年1月1日", "１０万円", "100万円",
    "５０００兆円", "1e10", "0x1F", "1/2", "1:30", "3-4", "1 000 000", "１ ０００", "ⅰ", "①",
    "²", "１０,０００", "9" * 20, "9" * 100, "一" * 50, "十" * 20, "万" * 20, "千百十", "百十",
    "十百", "1,000円", "約1,000人", "第1,234回", "3.5%", "-5", "+5", "1.0.0", "v1.2.3",
    "192.168.0.1", "2024.01.01", "1,2,3", "1, 2, 3", "1、2、3", "１，２，３", "０．０１",
    "一・二", "二十一世紀", "百二十三億四千五百万", "四十", "4十", "四10", "1十", "十1",
    "1万1千1百1十1", "〇.五", "1,000万", "1,000,000億", "1.5億", ".", ",", ",,", "..",
    "1,234,5678", "1,23,456", "12,3456,789", "1.2,345", "1,234.5,6", "0.000", "0,000",
    "1,000.", "1,000,", "1.000.000", "1．000", "1，000", "1。5",
]  # fmt: skip

KATAKANA_CASES = [
    "ヴァイオリン", "ァィゥェォ", "ッ", "ッッ", "ヵヶ", "ヷヸヹヺ", "ㇰㇱㇲ", "ヿ", "゠", "・",
    "ア・イ", "ヽヾ", "コーヒー", "コーヒーー", "ｱｲｳｴｵ", "ｱｲｳｴｵﾞ", "ァア", "アァ", "ーア",
    "アー", "ヮ", "ヮア", "ャュョ", "ャア", "アャ", "ンンン", "ヴヴヴ", "カ゛", "カ\u3099キ",
]  # fmt: skip

MULTI_CASES = [
    "…", "……", "…テスト…", "テスト…", "…テスト", "‥", "⋯", "㍿", "㍿㍿", "株式会社㍿です",
    "ﷺ", "ﷺ。", "ﷻ", "﷼", "㌀", "㌀㌀", "㌔㌘", "㋿", "㍻", "℡", "№", "㏍", "㈱", "½", "⅞",
    "ǅ", "ﬃ", "Ǆ", "ŉ", "ﾞ", "ｶﾞ", "⑽", "⒇", "⒈", "㉑", "🄁", "¨", "⑴", "㉐",
    "㏿", "ﷺﷻ", "".join(map(chr, range(0x3300, 0x3358))),
    "".join(map(chr, range(0x337B, 0x3380))), "ẛ\u0323", "ﬗ", "ﭏ", "ΐ",
    "a…b", "日…本", "… …", "…\n…", "。…。", "…。", "テスト……テスト", "「…」", "……。",
    "㍿\n㍿", "a㍿b", "ｶﾞ…", "…ｶﾞ", "…………",
]  # fmt: skip

EMOJI_CASES = [
    "👍", "👍🏽", "👨\u200d👩\u200d👧\u200d👦", "🇯🇵", "🇯🇵🇺🇸", "🏳\ufe0f\u200d🌈",
    "1\ufe0f\u20e3", "#\ufe0f\u20e3",
    "❤\ufe0f", "❤", "☺\ufe0f", "🏴\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",
    "😀😃😄", "テスト😀", "😀テスト", "😀 😀", "👩🏽\u200d💻", "🧑\u200d🤝\u200d🧑", "🫠", "🀄",
    "🃏", "🈁", "🈂\ufe0f", "🉐", "🆗", "🅰\ufe0f", "©\ufe0f", "™\ufe0f", "↔\ufe0f", "⌚",
    "\U0001f1e6", "🇦🇧🇨",
    "a👍b", "日👍本", "👍。", "🏻", "a🏻", "ア🏻", "👍\u200d", "\u200d👍", "🙇\u200d♂\ufe0f",
]  # fmt: skip

COMBINING_CASES = [
    "e\u0301", "e\u0301\u0302", "a\u0300\u0301\u0302\u0303", "\u0301", "\u0301\u0301",
    "日\u0301", "ア\u3099", "ア\u309a", "う\u3099", "は\u309a", "ｶ\u3099", "\u3099", "゛",
    "a\u20dd", "1\u20e3", "ก\u0e31", "क\u094dष", "\u1100\u1161",
    "\u1100\u1161\u11a8", "가\u11a8", "ñ", "n\u0303", "Å", "A\u030a", "\u212b", "ḱ",
    "q\u0307\u0323", "\u0344", "\u0340", "\u0f73", "क़", "⫝̸", "abc\u0301",
    "\u0301abc", "a\u200db", "a\u200cb", "ア\u200dイ", "葛\ufe00", "葛\U000e0100", "a\ufe0f",
    "1\ufe0f", "テ\u0308スト", "\u0301テスト", "テスト\u0301", " \u0301", "\n\u0301",
    "\u1ab0", "\u1dc0", "\ufe20", "a\u1ab0b", "ア\u20d0",
]  # fmt: skip

CONTROL_CASES = [
    "\x00", "a\x00b", "\x01\x02\x03", "\x07", "\x08", "\x1b[31mred\x1b[0m", "\x7f", "\x80",
    "\x9f", "\u200e", "\u200f", "\u202a\u202e", "\u2066\u2069", "\u00ad", "ソフト\u00adハイフン",
    "\u2061", "\u034f", "\u061c", "\ufff9\ufffa\ufffb", "￼", "�", "\U000e0001",
    "\U000e0020", "\u180b", "日\x00本", "テスト\x7fテスト", "\x00\x00\x00",
]  # fmt: skip

ODD_CASES = [
    "\ue000", "\uf8ff", "\U000f0000", "\U000ffffd", "\U00100000", "\U0010fffd", "a\ue000b",
    "日\ue000本", "\u0378", "\u0380", "\u05ff", "\u2fe0", "\U0002fffd", "\U000e0080",
    "\U000e01f0", "\U0003ffff", "\ufdd0", "\ufdef", "\ufffe", "\uffff", "\U0001fffe",
    "\U0001ffff", "\U0010fffe", "\U0010ffff", "a\uffffb", "日\ufffe本", "\U0010ffff\U0010ffff",
]  # fmt: skip

CATEGORY_CASES = [
    "abcαβγ", "αβγabc", "abcабв", "абвabc", "αβγабв", "àbç", "ÀÖØö", "×÷", "abc123",
    "123abc", "ａｂｃ１２３", "x²", "Ⅻ½", "日本語abc", "abc日本語", "ひらがなカタカナ",
    "カタカナひらがな", "漢字ひらがな", "一日", "日一", "日一〇〇", "〇〇日", "々々", "〆",
    "ヶ月", "ゝゞ", "゛゜", "ゟ", "＠＃％", "!?", "!!", "??", "$100", "€100", "100€", "¥100",
    "©®", "§¶", "←→", "∀∃", "⌘", "─━", "■□", "☀☁", "✓✗", "⠀⠁", "㈠", "㉠", "가나다",
    "ㄱㄴ", "\u1100", "ไทย", "עברית", "العربية", "हिन\u094dदी", "ελληνικά", "русский", "Ѐ", "ӹ",
    "Ӻ", "Ԑ", "ʹ", "ͻ", "ϻ", "ϼ", "ɐ", "ȷ", "ẚ", "ỹ", "Ỻ", "ⅿ", "ↀ", "↉", "⁰", "₉", "₠",
    "\u20d0", "℀", "⅏", "⺀", "⿕", "\u2fd6", "㐀", "䶵", "䶶", "鿿", "豈", "鶴", "侮", "龎", "﨎",
    "𠀀", "𪚲", "日本々", "々日本", "〇一二", "一〇〇〇", "三〇", "〇〇", "々〇", "〆〇",
    "αa", "aα", "αж", "жα", "a1", "1a", "αβγ123", "abc!", "!abc", "ア!", "!ア",
    "ひらがな123", "123ひらがな", "カタカナabc", "abcカタカナ", "a\u3000b", "１a", "a１",
    "ab\u0301c", "αβ\u0301γ", "日本\u3099",
]  # fmt: skip

MIXED_CASES = [
    "東京都に住んでいます。今日は\u3000いい天気ですね！  ２０２４年1,000円",
    "私はPythonとRustでプログラムを書きます。",
    "詳しくは https://example.com/path?q=日本語&x=1 を参照してください。",
    "メール: info@example.co.jp まで",
    '`print("こんにちは")` を実行',
    "def f(x):\n    return x * 2  # 日本語コメント\n",
    "<p>こんにちは<br/>世界</p>",
    '{"key": "値", "数": 123}',
    "C++とC#とF#",
    "iPhone 15 Pro Maxを買った",
    "COVID-19の影響で",
    "第3四半期の売上高は前年同期比12.5%増の1,234億円でした。",
    "「こんにちは」と彼は言った。『本当？』",
    "これは…テストです。。。本当に？！",
    "ｈｔｔｐｓ：／／ｅｘａｍｐｌｅ．ｃｏｍ",
    'Mr. Smith went to Tokyo. He said "Hello!" Then left.',
    "A.B.C. 株式会社",
    "１０：３０〜１２：００",
    "電話：03-1234-5678",
    "〒100-0001 東京都千代田区千代田1-1",
    "2024/01/01 12:34:56",
    "€1,234.56 ≒ 20万円",
    "#ハッシュタグ @mention",
    "ｗｗｗ",
    "草www",
    "(*´ω｀*)",
    "＼(^o^)／",
    "(´・ω・`)",
    "日本語\r\n改行\r\nテスト",
    "タブ\t区切り\tデータ",
    "  前後に空白  ",
    "\u3000\u3000全角インデント",
    "全角スペース\u3000区切り",
    "NHK・ＮＨＫ・ｎｈｋ",
    "Ⅲ章の①項",
    "ＪＲ東日本",
    "ﾃﾞｼﾞﾀﾙｶﾒﾗ",
    "ヴァージョン2.0",
    "スーパーーーマリオ",
    "マイナンバー（個人番号）",
    "漢字（かんじ）の読み方",
    "東京（とうきょう）",
    "東京(Tokyo)",
    "This is an English sentence. And another one!",
    "日本語と English が混ざった文章です。Really.",
    "ORTH",
    "LEMMA",
    "NOUN",
    "PROPN です",
    "IS_PUNCT",
    "LIKE_URL",
    "ENT_TYPE",
    "IS_ALPHA",
    "文。文",
    "文。」文",
    "文!文",
    "文?!文",
    "文...文",
    "文。\n文",
    "。",
    "。。",
    "文。 文",
    "文。\u3000文",
    "文\n\n文",
    "文。」「文",
    "文！？文",
    "文．文",
    "文｡文",
    "Hello. World.",
    "Hello.World",
    "e.g. this",
    "終わり。 ",
    " 始まり。",
    "。始まり",
]

SPACY_SYMBOLS = ["IS_LOWER", "LENGTH", "PUNCT", "ADJ", "VERB", "SPACY", "PROB", "LANG"]


def case_items(fineweb_texts: Sequence[str]) -> list[Item]:
    rng = random.Random(f"{SEED}/cases")
    ignore, replace = rewrite_def()
    out: list[Item] = []

    def add(kind: str, *texts: str) -> None:
        out.extend(({"k": kind}, t) for t in texts)

    add("empty", "")
    for i in range(0, len(ignore), 8):
        add("rw-ignore", "".join(ignore[i : i + 8]))
    add("rw-ignore", *("\n".join(f"日{c}本" for c in ignore[i : i + 8]) for i in range(0, 64, 8)))
    add("rw-ignore", *("A" + c + "B" for c in rng.sample(ignore, 16)))
    for i in range(0, len(replace), 8):
        group = replace[i : i + 8]
        add("rw-replace", "".join(group), "\n".join(f"これは{k}です" for k in group))
    add("rw-replace", *(k[0] + " " + k[1:] for k in replace[::6]))
    add("rw-replace", *("A" + k + k for k in replace[::9]))
    add("nfkc", *NFKC_CASES)
    add("case", *CASE_CASES)
    add("psm", *PSM_CASES)
    add("psm", *(m * n for m in ("ー", "-", "⁓", "〜", "〰", "ｰ", "－") for n in (1, 2, 3, 6)))

    pairs = [("(", ")"), ("（", "）"), ("(", "）"), ("（", ")")]
    for (lb, rb), base, reading in itertools.product(
        pairs, ("漢字", "東京"), ("かんじあいう", "カンジアイウ", "かンじーアい")
    ):
        add("yomi", *(f"{base}{lb}{reading[:n]}{rb}です" for n in range(1, 7)))
    add(
        "yomi",
        "漢字(か(ん)じ)", "漢字((かんじ))", "漢字(かんじ", "漢字かんじ)", "(かんじ)", "あ(かんじ)",
        "漢字()", "漢字( かんじ)", "漢字(かん じ)", "漢字(ｶﾝｼﾞ)", "漢字(ｶﾝｼﾞｱｲ)", "漢字(漢字)",
        "漢字(abc)", "漢字(かんじ)(かんじ)", "日本(にほん)語(ご)", "〆(しめ)", "々(のま)",
        "〇(まる)", "漢字[かんじ]", "漢字【かんじ】", "ア(あ)", "漢字(ア)", "漢字(ーー)",
        "漢字(かーーーん)", "漢字(か)ー", "漢字(かんじ)漢字(かんじ)", "漢(か)", "漢字(ゝ)",
        "漢字(ゟ)", "漢字(ヿ)", "漢字(゛)", "漢字(か\u3099)", "一(いち)", "二〇(にじゅう)",
        "漢字 (かんじ)", "漢字(\u3000かんじ)", "漢字(か\nんじ)", "𠀀(かんじ)", "⺀(かんじ)",
    )  # fmt: skip

    add("num", *NUMERIC_CASES)
    add("num", *(f"約{n}円です" for n in NUMERIC_CASES[::3]))

    add("kata", *KATAKANA_CASES)
    kana = [chr(c) for c in range(0x30A1, 0x30F7)]
    half = [chr(c) for c in range(0xFF66, 0xFF9E)]
    for n in range(1, 7):
        for _ in range(3):
            s = "".join(rng.choice(kana) for _ in range(n))
            h = "".join(rng.choice(half) for _ in range(n))
            add("kata", s, "東京" + s, s + "です", "ラーメン" + s, s + "ー", "ァ" + s, s + "ッ")
            add("kata", "ヽ" + s, "テスト" + s + "テスト", h, "東京" + h + "です", s + h)

    reps = category_representatives()
    for a, _ in reps:
        add("cat", "".join(a + b + "\n" for b, _ in reps))
    triples = [
        a + b + c + c
        for (a, ca), (b, cb), (c, cc) in itertools.product(reps, repeat=3)
        if meets(ca, cb) and meets(cb, cc) and not meets(ca, cc)
    ]
    picked = rng.sample(triples, min(64, len(triples)))
    add("cat", *("\n".join(picked[i : i + 8]) for i in range(0, len(picked), 8)))
    add("cat", *CATEGORY_CASES)

    for w in WHITESPACE:
        add("ws", w, w * 2, f"日本{w}語", f"a{w}b", f"{w}日本", f"日本{w}", f"テスト{w}{w}テスト")
        add("ws", f"{w}a{w}", f"文。{w}文", f"word.{w}Next")
    for n in range(1, 5):
        sp, wide = " " * n, "\u3000" * n
        add("spaces", f"日本{sp}語", f"a{sp}b", sp, f"{sp}テスト", f"テスト{sp}", f"{sp}テスト{sp}")
        add("spaces", f"テスト{sp}。{sp}次", wide, f"日本{wide}語", "\t" * n)
        add("spaces", "\n" * n, " \n" * n, f"a{sp}\n{sp}b", f"word.{sp}Next", f"日本。{sp}次")
        add("spaces", f"「{sp}」", f"({sp})", f"{sp}a", f"a{sp}", f"1{sp}2", f"テスト{sp}test")
    add(
        "spaces",
        "テスト \u3000 テスト", "a \u3000b", "a\u3000 b", " \u3000", "\u3000 ", " \t ", "\t \t",
        " \n ", "\n \n", " \r\n ", "\r\n\r\n", "a \t b", "テスト\u3000\u3000テスト", "  \u3000  ",
        "a\u00a0 b", "a \u00a0b", "日本 \x0c 語", "a\x1c b", " \x85 ", "a\u2028 b",
    )  # fmt: skip

    add("emoji", *EMOJI_CASES)
    add("comb", *COMBINING_CASES)
    add("ctrl", *CONTROL_CASES)
    add("odd", *ODD_CASES)
    add("multi", *MULTI_CASES)
    add("mixed", *MIXED_CASES, *SPACY_SYMBOLS)
    add("mixed", *(f"これは{s}です。" for s in SPACY_SYMBOLS))

    runs = ["ア", "a", "1", " ", "１", "\u3000", "あ", "漢", "ー", "-", "…", "。", "\n", "ｱ"]
    runs += ["😀", "é", "e\u0301", "ab", "アイ", "1,", "1,000", "ｶﾞ", "㍿", "日本", "テスト "]
    runs += ["\r\n", "\t", "a ", "ア ", "  ", "、", "!", "ﾞ", "〜", "x\u3099", "Ａ", "Ж", "α"]
    for r in runs:
        add("long", r * max(1, 300 // len(r)), r * max(1, 10_000 // len(r)))
    # Long runs over 10000 characters that still fit in 40000 bytes.
    add("long", "ア" * 13_333, "a" * 40_000, "1" * 40_000, " " * 40_000, "😀" * 10_000)

    add(
        "limit",
        "a" * MAX_INPUT,
        "a" * (MAX_INPUT + 1),
        "ア" * (MAX_INPUT // 3),
        "ア" * (MAX_INPUT // 3) + "a",
        "𠀀" * 12_287 + "a",
        "𠀀" * 12_288,
        "é" * 24_574 + "a",
        "é" * 24_575,
        "a" * (MAX_INPUT - 4) + "😀",
        "a" * (MAX_INPUT - 3) + "😀",
        "1" * MAX_INPUT,
        " " * MAX_INPUT,
        "ｶﾞ" * 8191,
        "ｶﾞ" * 8192,
        # NFKC turns "﷼" (3 bytes) into "ریال" (8 bytes), "㍿" (3) into "株式会社" (12) and
        # "ﷺ" (3) into "صلى الله عليه وسلم" (33): limits on the edited input.
        "﷼" * (MAX_EDITED // 8) + "a" * (MAX_EDITED % 8),
        "﷼" * (MAX_EDITED // 8) + "a" * (MAX_EDITED % 8 + 1),
        "﷼" * (MAX_EDITED // 8 + 1),
        "﷼" * 13_000,
        "﷼" * 20_000,
        "﷼" * (MAX_INPUT // 3),
        "㍿" * (MAX_EDITED // 12),
        "㍿" * (MAX_EDITED // 12 + 1),
        "ﷺ" * (MAX_EDITED // 33),
        "ﷺ" * (MAX_EDITED // 33 + 1),
        "…" * (MAX_INPUT // 3),
        "ア" * 10_000 + "﷼" * 4000,
    )

    # FineWeb-2 lines in odd contexts.
    lines = [x for t in fineweb_texts[:40] for x in t.splitlines() if 8 <= len(x) <= 80]
    for line in rng.sample(lines, min(24, len(lines))):
        add("fw-ctx", f"  {line}  ", line.replace("。", "。 "), f"ｶﾞ{line}…", line + "\u3000" * 3)
    return clean(out)


# --------------------------------------------------------------------------
# Inputs: FineWeb-2, fuzz, sweep, long documents


def read_fineweb(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=["text"]).column("text").to_pylist()


def fineweb_source(path: Path, n: int) -> dict[str, Any]:
    return {"file": path.name, "sha256": sha256_file(path), "rows": n, "dataset": FINEWEB_DATASET}


def split_bytes(s: str, limit: int) -> list[str]:
    """`s` cut at character boundaries into pieces of at most `limit` bytes."""
    out = []
    while utf8len(s) > limit:
        head = s.encode("utf-8")[:limit].decode("utf-8", errors="ignore")
        out.append(head)
        s = s[len(head) :]
    out.append(s)
    return out


def line_items(docs: Sequence[str]) -> list[Item]:
    items: list[Item] = []
    for d, doc in enumerate(docs):
        for ln, line in enumerate(doc.splitlines(keepends=True)):
            pieces = split_bytes(line, DT_CHUNK)
            for p, piece in enumerate(pieces):
                meta: dict[str, Any] = {"d": d, "l": ln}
                if len(pieces) > 1:
                    meta["p"] = p
                items.append((meta, piece))
    return clean(items)


def sample_lines(items: Sequence[Item]) -> list[Item]:
    first: dict[str, int] = {}
    for i, (_, text) in enumerate(items):
        first.setdefault(text, i)
    rng = random.Random(f"{SEED}/fineweb-lines")
    return [items[i] for i in sorted(rng.sample(sorted(first.values()), N_FINEWEB_LINES))]


def doc_items(docs: Sequence[str]) -> list[Item]:
    return clean([({"d": d}, doc) for d, doc in enumerate(docs)])


def sample_docs(items: Sequence[Item]) -> list[Item]:
    rng = random.Random(f"{SEED}/fineweb-docs")
    fits = [i for i, (_, text) in enumerate(items) if utf8len(text) <= 8000]
    return [items[i] for i in sorted(rng.sample(fits, N_FINEWEB_DOCS))]


def sweep_items(sample: bool) -> list[Item]:
    bases = [b for b in range(0, 0x110000, SWEEP_WIDTH) if not 0xD800 <= b < 0xE000]
    kinds = {}
    for kind in ("run", "ctx"):
        if sample:
            rng = random.Random(f"{SEED}/sweep-{kind}")
            bmp = [b for b in bases if b < 0x10000]
            rest = [b for b in bases if b >= 0x10000]
            kinds[kind] = sorted(rng.sample(bmp, N_SWEEP_BMP) + rng.sample(rest, N_SWEEP_REST))
        else:
            kinds[kind] = bases
    items: list[Item] = [
        ({"k": "run", "cp": b}, "".join(map(chr, range(b, b + SWEEP_WIDTH)))) for b in kinds["run"]
    ]
    items += [
        ({"k": "ctx", "cp": b}, "\n".join(f"日{chr(c)}本" for c in range(b, b + SWEEP_WIDTH)))
        for b in kinds["ctx"]
    ]
    return clean(items)


def fuzz_items(fineweb_texts: Sequence[str]) -> list[Item]:
    rng = random.Random(f"{SEED}/fuzz")
    _, replace = rewrite_def()
    freq = Counter(c for t in fineweb_texts for c in t if "一" <= c <= "鿿")
    kanji = [c for c, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:400]]
    pools: dict[str, list[str]] = {
        "hira": [chr(c) for c in range(0x3041, 0x3097)],
        "kata": [chr(c) for c in range(0x30A1, 0x30FB)] + ["ー"],
        "half": [chr(c) for c in range(0xFF61, 0xFFA0)],
        "kanji": kanji,
        "knum": list("〇一二三四五六七八九十百千万億兆"),
        "ascii": [chr(c) for c in range(0x30, 0x7B) if chr(c).isalnum()],
        "punct": list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"),
        "full": [chr(c) for c in range(0xFF01, 0xFF5F)],
        "ws": WHITESPACE,
        "psm": ["ー", "-", "⁓", "〜", "〰", "ｰ", "－", "~", "～"],
        "brk": list("()（）[]「」『』【】"),
        "comb": ["\u0301", "\u0308", "\u3099", "\u309a", "\ufe0f", "\u200d", "\u20e3", "🏽"],
        "emoji": ["😀", "👍", "🇯🇵", "👨\u200d👩\u200d👧", "❤\ufe0f", "🀄", "1\ufe0f\u20e3"],
        "greek": [chr(c) for c in range(0x391, 0x3CA) if unicodedata.category(chr(c)) != "Cn"],
        "cyr": [chr(c) for c in range(0x400, 0x460)],
        "latin": [chr(c) for c in range(0xC0, 0x180)],
        "rw": replace,
        "multi": list("…㍿①Ⅻﷺﬁ½㌀㋿℡‥⑴ǅ"),
        "sym": list("、。・「」『』〈〉《》※◎●○■□△▲▽▼◇◆★☆→←↑↓"),
        "odd": ["\x00", "\x7f", "\u200b", "\ufeff", "\ue000", "\ufffe", "\u0378", "\U0010ffff"],
    }
    names = [*pools, "fineweb", "fineweb"]
    items: list[Item] = []
    while len(items) < N_FUZZ:
        pieces = []
        for _ in range(rng.randint(1, 5)):
            name = rng.choice(names)
            if name == "fineweb":
                doc = rng.choice(fineweb_texts)
                at = rng.randrange(len(doc))
                piece = doc[at : at + rng.randint(1, 10)]
            else:
                piece = "".join(rng.choice(pools[name]) for _ in range(rng.randint(1, 3)))
            if rng.random() < 0.08:
                piece *= rng.randint(2, 4)
            pieces.append(piece)
        text = "".join(pieces)[:40]
        if SKIP not in text:
            items.append(({}, text))
    return items


def long_items(fineweb_texts: Sequence[str]) -> list[Item]:
    """Documents over 40000 bytes, several with a chunk boundary at a chosen character."""
    from datatrove.utils.word_tokenizers import chunk_text_on_bytes

    def cuts(text: str) -> set[int]:
        return set(itertools.accumulate(len(c) for c in chunk_text_on_bytes(text, DT_CHUNK)))

    items: list[Item] = []

    def search(name: str, build: Callable[..., tuple[str, list[int]]], *ranges: range) -> None:
        for params in itertools.product(*ranges):
            text, wanted = build(*params)
            if set(wanted) <= cuts(text):
                items.append(({"k": name}, text))
                return
        raise AssertionError(f"no chunk boundary where {name} needs one")

    def fill(head: Callable[[int], str], budget: int) -> int:
        """The largest k with utf8len(head(k)) <= budget."""
        lo, hi = 0, budget
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if utf8len(head(mid)) <= budget:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def place(name: str, pad: str, probe: str, at: int, tail: str = "イ" * 3000 + "。") -> None:
        """x * extra + pad * k + probe + tail, with a chunk boundary `at` characters into `probe`.

        The ASCII prefix moves the chunker's first guess (it scales with the ratio of
        characters to bytes), so that some k and extra put a boundary there.
        """
        k0 = fill(lambda k: pad * k, DT_CHUNK - utf8len(probe[:at]))

        def build(k: int, extra: int) -> tuple[str, list[int]]:
            return "x" * extra + pad * k + probe + tail, [extra + k * len(pad) + at]

        search(name, build, range(k0, max(k0 - 64, 0), -1), range(40))

    for probe, at in [
        ("テスト ", 4), (" word ", 6), ("word ", 5), ("word  ", 5), ("word  ", 6),
        ("word   ", 5), ("word   ", 6), ("テスト\u3000", 4), ("テスト\n", 4), ("テスト\n\n", 4),
        ("テスト\r\n", 4), ("テスト。", 4), ("テスト。 ", 5), ("テスト。", 3), ("ｶﾞ", 1),
        ("e\u0301", 1), ("👨\u200d👩\u200d👧", 2), ("1,000", 2), ("--", 1), ("漢字(かんじ)", 3),
        ("テスト", 1), ("…", 1), ("…テスト", 1), ("  ", 1), (" " * 10, 5), ("\u3000\u3000", 1),
        ("テスト \u3000", 4), (" テスト", 1), ("テスト 。", 4), ("! ", 2),
    ]:  # fmt: skip
        place(f"cut {probe!r} at {at}", "ア", probe, at)
    place("cut after ' ' in ASCII", "a", "  word ", 7, "b" * 3000 + ".")
    place("cut inside '  ' in ASCII", "a", " ", 1, " " + "b" * 3000 + ".")
    place("cut inside 'x  ' in ASCII", "a", "x  ", 2, "b" * 3000)
    place("cut after ' ' in digits", "1", " ", 1, "2" * 3000)
    place("cut after 'テスト ' in emoji", "😀", "テスト ", 4, "😃" * 3000)
    place("cut after ' word ' in Latin-1", "é", " word ", 6, "è" * 3000)
    # A last chunk of whitespace only (the chunker cannot cut fewer than 10 characters from
    # the end unless the cut falls at exactly 40000 bytes).
    place("cut before 16 trailing spaces", "ア", " " * 16, 0, "")
    place("cut before 16 trailing newlines", "ア", "\n" * 16, 0, "")
    place("cut before 16 trailing ideographic spaces", "ア", "　" * 16, 0, "")

    # A boundary after "テスト " in both cuts of a three-chunk document.
    k0 = fill(lambda k: "ア" * k, DT_CHUNK - utf8len("テスト "))

    def three(k: int, m: int) -> tuple[str, list[int]]:
        text = "ア" * k + "テスト " + "ウ" * m + "テスト " + "エ" * 500
        return text, [k + 4, k + 4 + m + 4]

    m0 = fill(lambda m: "ウ" * m, DT_CHUNK - utf8len("テスト "))
    search("two cuts after 'テスト '", three, range(k0, k0 - 24, -1), range(m0 + 4, m0 - 40, -1))

    fw = "\n".join(fineweb_texts)
    for probe, at in [("テスト ", 4), (" word ", 6), ("word  ", 5), ("。", 1)]:
        k0 = fill(lambda k: fw[:k], DT_CHUNK - utf8len(probe[:at]))

        def build_fw(k: int, extra: int, probe: str = probe, at: int = at) -> tuple[str, list[int]]:
            return fw[:k] + probe + fw[k : k + 2000] + "x" * extra, [k + at]

        search(f"FineWeb-2, cut {probe!r} at {at}", build_fw, range(k0, k0 - 24, -1), range(40))

    def add(name: str, text: str) -> None:
        items.append(({"k": name}, text))

    add("whitespace-only middle chunk", "ア" * 13_000 + " " * 41_000 + "イ" * 100)
    add("ideographic spaces", "ア" * 13_000 + "\u3000" * 14_000 + "イ" * 100)
    add("newlines", "ア" * 13_000 + "\n" * 41_000 + "イ" * 100)
    add("exactly 40000 bytes", "ア" * 13_333 + "a")
    add("40001 bytes", "ア" * 13_333 + "ab")
    add("a * 40001", "a" * 40_001)
    add("emoji, 40000 bytes", "😀" * 10_000)
    add("emoji, 40001 bytes", "😀" * 10_000 + "a")
    add("katakana", "ア" * 50_000)
    add("ASCII letters", "a" * 100_000)
    add("digits", "1" * 50_000)
    add("spaces", " " * 50_000)
    add("ideographic spaces only", "\u3000" * 20_000)
    add("kanji", "漢" * 20_000)
    add("hiragana", "あ" * 20_000)
    add("rial sign", "﷼" * 20_000)
    sentence = "The quick brown fox jumps over the lazy dog. "
    add("English", sentence * (50_000 // len(sentence)))
    add("words", "word " * 10_000)
    mixed = "東京でPythonを使う。詳しくは https://example.com/日本 を見て！ 1,000円です。\n"
    add("mixed", mixed * (60_000 // utf8len(mixed) + 1))
    docs, size = [], 0
    for limit in (45_000, 90_000, 130_000):
        while size <= limit:
            docs.append(fineweb_texts[len(docs)])
            size += utf8len(docs[-1]) + 1
        add(f"FineWeb-2 documents, over {limit} bytes", "\n".join(docs))
    add("FineWeb-2 documents joined by spaces", " ".join(fineweb_texts[:60])[:50_000])
    return clean(items)


# --------------------------------------------------------------------------
# chunk_text_on_bytes inputs

CHUNK_CHARS = {1: "a Z0.", 2: "éñαжש", 3: "あア漢。ー", 4: "😀𠀀𝐀🀄"}


def chunk_text(rng: random.Random, n: int, size: int) -> str:
    """A text of exactly `size` UTF-8 bytes: a few runs of one character, some single ones."""
    if size <= 64:
        out = []
        left = size
        while left > 0:
            w = min(rng.randint(1, 4), left)
            run = min(left // w, rng.randint(1, 4))
            out.append(rng.choice(CHUNK_CHARS[w]) * run)
            left -= run * w
        return "".join(out)
    modes = ["homog", "periodic"] if n >= 1_000_000 else ["homog", "mixed", "front", "periodic"]
    mode = rng.choice(modes)
    if mode == "periodic":
        pattern = "".join(
            rng.choice(CHUNK_CHARS[rng.randint(1, 4)]) for _ in range(rng.randint(1, 6))
        )
        text = pattern * (size // utf8len(pattern))
        return "a" * (size - utf8len(text)) + text
    if mode == "homog":
        widths = [rng.randint(1, 4)] * rng.randint(1, 3)
    elif mode == "front":  # one width, then another
        first, second = rng.sample([1, 2, 3, 4], 2)
        widths = [first, second]
    else:
        widths = [rng.randint(1, 4) for _ in range(rng.randint(2, 6))]
    singles = (
        [] if mode == "homog" or rng.random() < 0.5 else [rng.randint(1, 4)] * rng.randint(1, 3)
    )
    budget = size - sum(singles)
    cuts = sorted(rng.randint(0, budget) for _ in range(len(widths) - 1))
    parts = [b - a for a, b in itertools.pairwise([0, *cuts, budget])]
    runs: list[str] = []
    left = 0
    for w, part in zip(widths, parts, strict=True):
        runs.append(rng.choice(CHUNK_CHARS[w]) * ((part + left) // w))
        left = (part + left) % w
    for w in singles:
        runs.insert(rng.randint(0, len(runs)), rng.choice(CHUNK_CHARS[w]))
    return "a" * left + "".join(runs)


def chunk_rows() -> list[tuple[int, str]]:
    rng = random.Random(f"{SEED}/chunks")
    rows: list[tuple[int, str]] = []
    for n, count in CHUNK_PLAN.items():
        top = 3 * n if n >= 40_000 else 5 * n
        special = [0, 1, 2, 3, 4, 5, n - 3, n - 2, n - 1, n, n + 1, n + 2, n + 3, n + 4]
        special += [2 * n - 1, 2 * n, 2 * n + 1, 3 * n - 1, 3 * n, 3 * n + 1]
        for j in range(count):
            size = special[j % len(special)] if j < 3 * len(special) else rng.randint(0, top)
            rows.append((n, chunk_text(rng, n, max(size, 0))))
    return rows


# --------------------------------------------------------------------------
# Commands


def out_dir(args: argparse.Namespace) -> Path:
    out: Path = args.out
    if getattr(args, "full", False) and out.resolve() == HERE:
        sys.exit("--full writes large files: give an --out directory outside the repository")
    return out


def cmd_tags(args: argparse.Namespace) -> None:
    import sudachipy
    from spacy.lang.ja.tag_map import TAG_MAP

    dic = sudachipy.Dictionary()
    rows = []
    for i in itertools.count():
        pos = dic.pos_of(i)
        if pos is None:
            break
        tag = "-".join([x for x in pos[:4] if x != "*"])
        rows.append([i, list(pos), tag, tag in TAG_MAP])
    missing = [r[0] for r in rows if not r[3]]
    for r in rows:
        if not r[3]:
            log(f"POS {r[0]} {r[1]}: tag {r[2]!r} is not in TAG_MAP")
    path = out_dir(args) / "tags.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    head = header("tags", "tags", "full", rows=len(rows))
    body = ",\n".join(dumps(r) for r in rows)
    path.write_text(
        f'{{"header":{dumps(head)},\n"missing":{dumps(missing)},\n"pos":[\n{body}\n]}}\n',
        encoding="utf-8",
    )
    log(f"wrote {path} ({len(rows)} POS, {len(missing)} tags missing from TAG_MAP)")


def cmd_chunks(args: argparse.Namespace) -> None:
    from datatrove.utils.word_tokenizers import chunk_text_on_bytes

    rows = chunk_rows()
    lines = [""] * len(rows)
    order = range(len(rows) - 1, -1, -1) if args.reverse else range(len(rows))
    t0 = time.monotonic()
    for i in order:
        n, text = rows[i]
        row: dict[str, Any] = {"t": pack(text), "n": n}
        try:
            row["c"] = [len(c) for c in chunk_text_on_bytes(text, n)]
        except Exception as e:
            row["x"] = err(e)
        lines[i] = dumps(row)
    log(f"chunks: {len(rows)} rows in {time.monotonic() - t0:.1f} s")
    write_jsonl(out_dir(args) / "chunks.jsonl", header("chunks", "chunks", "full"), lines)


def run_level(args: argparse.Namespace, level: str) -> None:
    out = out_dir(args)
    full = args.full
    selection = "full" if full else "committed"
    docs = read_fineweb(args.fineweb)
    src = fineweb_source(args.fineweb, len(docs))
    record = morpheme_recorder() if level == "morphemes" else datatrove_recorder()

    def emit(generator: str, items: Sequence[Item], **extra: Any) -> None:
        lines = record_items(items, record, args.reverse, f"{level}-{generator}")
        if not full:
            lines = small(lines)
        head = header(level, generator, selection, **extra)
        write_jsonl(out / f"{level}-{generator}.jsonl", head, lines)

    emit("cases", case_items(docs), source=src)
    if level == "morphemes":
        lines = line_items(docs)
        emit("fineweb", lines if full else sample_lines(lines), source=src)
    else:
        whole = doc_items(docs)
        emit("fineweb", whole if full else sample_docs(whole), source=src)
    emit("fuzz", fuzz_items(docs), source=src)
    emit("sweep", sweep_items(sample=not full))
    if level == "datatrove":
        emit("long", long_items(docs), source=src)


def cmd_large(args: argparse.Namespace) -> None:
    out = out_dir(args)
    docs = read_fineweb(args.parquet)
    src = fineweb_source(args.parquet, len(docs))
    log(f"{args.parquet}: {len(docs)} documents, {sum(map(len, docs))} characters")
    for level, generator, items in (
        ("morphemes", "large-lines", line_items(docs)),
        ("datatrove", "large-docs", doc_items(docs)),
    ):
        record = morpheme_recorder() if level == "morphemes" else datatrove_recorder()
        chars = sum(len(t) for _, t in items)
        t0 = time.monotonic()
        lines = record_items(items, record, args.reverse, f"{level}-{generator}")
        took = time.monotonic() - t0
        log(f"{level}: {len(items)} rows, {chars} characters, {took:.1f} s")
        head = header(level, generator, "full", source=src, characters=chars)
        write_jsonl(out / f"{level}-{generator}.jsonl", head, lines)


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
    # datatrove's tokenizer calls the deprecated Dictionary.create().
    warnings.filterwarnings("ignore", message=r"Dictionary\.create\(\) is deprecated")
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, helptext in (
        ("tags", "every POS of the dictionary and its spaCy tag"),
        ("chunks", "datatrove's chunk_text_on_bytes on random texts"),
        ("morphemes", "SudachiPy's morphemes (split mode A)"),
        ("datatrove", "datatrove's word, span and sentence tokenization"),
    ):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--out", type=Path, default=HERE)
        if name != "tags":
            s.add_argument("--reverse", action="store_true", help="process rows in reverse order")
        if name in ("morphemes", "datatrove"):
            s.add_argument("--fineweb", type=Path, default=FINEWEB)
            s.add_argument("--full", action="store_true", help="every row, not the committed ones")
    s = sub.add_parser("large", help="every line and document of a FineWeb-2 parquet file")
    s.add_argument("parquet", type=Path)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--reverse", action="store_true")
    s = sub.add_parser("compare", help="check that two directories hold identical recordings")
    s.add_argument("a", type=Path)
    s.add_argument("b", type=Path)
    args = p.parse_args(argv)
    commands: dict[str, Callable[[argparse.Namespace], None]] = {
        "tags": cmd_tags,
        "chunks": cmd_chunks,
        "morphemes": lambda a: run_level(a, "morphemes"),
        "datatrove": lambda a: run_level(a, "datatrove"),
        "large": cmd_large,
        "compare": cmd_compare,
    }
    commands[args.cmd](args)


if __name__ == "__main__":
    main()
