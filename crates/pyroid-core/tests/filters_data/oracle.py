"""Record what datatrove 0.10.1 returns for word and sentence splitting.

Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace, spaCy 3.8.18
(MIT), © ExplosionAI GmbH, and, for the Indic fuzzer's word lists,
indic-nlp-library 0.92 (MIT), © Anoop Kunchukuttan, and urduhack 1.0.3 (MIT),
© Urdu Hack; modified.

datatrove's `split_into_words` / `split_into_sentences` (default language
"eng") run spaCy 3.8.18 `spacy.blank("en")` plus the `sentencizer` pipe. The
Rust port (`pyroid_core::filters::tokenizer`) must give identical results; this
script records the oracle and `tokenizer_test.rs` replays it.

`--lang CODE` records any other datatrove language code instead (`deu_Latn`,
`kat_Geor`, `hin_Deva`, ...): datatrove picks the tokenizer from its
`tokenizer_assignment.csv` (a spaCy blank model, the WhitespaceTokenizer, the
IndicNLPTokenizer, ...) and the fuzzer aims at that language's rules and scripts.

Run on Linux (or in a Linux container), with a venv holding
datatrove==0.10.1, spacy==3.8.18, regex and pyarrow (plus indic-nlp-library for
the Indic codes):

    python oracle.py golden OUT.jsonl                # the committed English recording
    python oracle.py --lang deu_Latn golden OUT.jsonl  # a committed per-language one
    python oracle.py goldens DIR [KEY ...]           # every multilingual golden-KEY.jsonl
    python oracle.py corpus OUT.jsonl --source dev|100k|adversarial|fineweb2|fw2:SLICE
    python oracle.py fuzz OUT.jsonl N SEED           # N generated strings
    python oracle.py unicode OUT.jsonl N SEED        # N strings aimed at Unicode versions
    python oracle.py rerecord IN.jsonl OUT.jsonl     # IN's texts, on this interpreter
    python oracle.py isalpha [OUT.jsonl]             # the IS_ALPHA history effect
    python oracle.py langtable OUT.json --distribution CSV  # FineWeb-2 volume per tokenizer

Each output line is one text, recorded with one call of each datatrove function:

    {"t": text, "w": split_into_words(text, lang), "s": len(split_into_sentences(text, lang)),
     "u": unicodedata.unidata_version}

or, when datatrove raises, {"t": ..., "raise": "ExceptionType", "at": "w"|"s", "u": ...}.
With `--lang` other than "eng" every row also carries "l": the language code, and
a row that raised in `split_into_sentences` ("at": "s") still carries "w".
Rows may also carry "k" (what the text is: "doc", "line", "excerpt", "edge" or the
fuzz generator's name) and "id" (the source document). Raw U+0085, U+2028 and
U+2029 are written as JSON escapes so that every line splitter agrees.

The fuzz inputs draw on the running interpreter's `unicodedata`, so one seed
gives slightly different texts on different CPython versions; `rerecord` replays
the exact texts of an existing recording instead.
"""

# The generator is full of look-alike characters on purpose.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import csv
import functools
import glob
import heapq
import itertools
import json
import multiprocessing
import random
import re
import string
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from re import _constants as sre_c
from re import _parser as sre_p
from typing import Any

UNIDATA = unicodedata.unidata_version
# The datatrove language code passed to split_into_words / split_into_sentences (--lang).
LANG = "eng"
FW2_DIR = Path("/data/fineweb2")
SOURCES: dict[str, tuple[list[str], bool]] = {
    # name: (parquet files, also record C4-style lines)
    "dev": (["/data/fineweb/fineweb-dev-10k.parquet"], True),
    "100k": (["/data/fineweb/fineweb-100k.parquet"], False),
    "adversarial": (["/data/parity/adversarial.parquet"], True),
    "fineweb2": (sorted(glob.glob("/data/fineweb2/*.parquet")), False),
    # one FineWeb-2 slice, e.g. fw2:deu_Latn (record it with --lang deu_Latn)
    **{f"fw2:{Path(p).stem}": ([p], True) for p in sorted(glob.glob(f"{FW2_DIR}/*.parquet"))},
}
# datatrove.pipeline.filters.c4_filters.CITATION_REGEX
CITATION = re.compile(r"\[\d*]|\[edit]|\[citation needed]")
GOLDEN_SEED = 20261105
GOLDEN_FUZZ = 3000

Item = tuple[str, dict[str, Any]]


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


# --------------------------------------------------------------------------
# Recording


def record(text: str) -> dict[str, Any]:
    """One oracle row: datatrove's words and sentence count for `text` in `LANG`."""
    from datatrove.utils.text import split_into_sentences, split_into_words

    row: dict[str, Any] = {"t": text}
    at = "w"
    try:
        row["w"] = list(split_into_words(text, LANG))
        at = "s"
        n = len(split_into_sentences(text, LANG))
    except Exception as e:
        if LANG == "eng":  # the English recordings' format: no words in a row that raised
            row.pop("w", None)
        row["raise"] = type(e).__name__
        row["at"] = at
    else:
        row["s"] = n
    row["u"] = UNIDATA
    if LANG != "eng":
        row["l"] = LANG
    return row


def dumps(row: dict[str, Any]) -> str:
    s = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    # Only possible inside JSON strings, where the escapes mean the same.
    return s.replace("\x85", "\\u0085").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _job(item: Item) -> tuple[str, str | None]:
    text, extra = item
    row = record(text)
    row.update(extra)
    return dumps(row), row.get("raise")


def _blocks(items: Iterable[Item], size: int) -> Iterator[list[Item]]:
    it = iter(items)
    while block := list(itertools.islice(it, size)):
        yield block


def recorded(
    items: Iterable[Item], workers: int, chunksize: int
) -> Iterator[tuple[str, str | None]]:
    """(JSON line, exception name or None) for every item, in input order, from a fork pool."""
    record("Warm up. Load spaCy before forking.")
    ctx = multiprocessing.get_context("fork")
    # Fork before reading any parquet, so no pyarrow threads exist at fork time.
    with ctx.Pool(workers) as pool:
        for block in _blocks(items, 20_000):
            yield from pool.imap(_job, block, chunksize)


def run(items: Iterable[Item], path: Path, workers: int, chunksize: int) -> None:
    """Record every item with a fork pool, in input order, streaming to `path`."""
    t0 = time.monotonic()
    raises: Counter[str] = Counter()
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for line, raised in recorded(items, workers, chunksize):
            out.write(line + "\n")
            n += 1
            if raised:
                raises[raised] += 1
    report(path, n, raises, t0)


def report(path: Path, n: int, raises: Counter[str], t0: float) -> None:
    dt = time.monotonic() - t0
    lang = "" if LANG == "eng" else f"--lang {LANG}, "
    log(
        f"wrote {path}: {n} rows, {lang}Unicode {UNIDATA}, Python {sys.version.split()[0]}, "
        f"{dt:.1f}s, {path.stat().st_size / 1e6:.2f} MB, raised: {dict(raises)}"
    )


# --------------------------------------------------------------------------
# Real documents


def read_docs(source: str, limit: int | None = None) -> Iterator[tuple[str, str]]:
    """(text, id) for each document of `source`, in file order."""
    import pyarrow.parquet as pq

    n = 0
    for p in SOURCES[source][0]:
        pf = pq.ParquetFile(p)
        has_id = "id" in pf.schema_arrow.names
        stem = Path(p).stem
        cols_wanted = ["text", "id"] if has_id else ["text"]
        for batch in pf.iter_batches(batch_size=1024, columns=cols_wanted, use_threads=False):
            cols = batch.to_pydict()
            ids = cols["id"] if has_id else [None] * len(cols["text"])
            for text, doc_id in zip(cols["text"], ids, strict=True):
                if limit is not None and n >= limit:
                    return
                if text is None:
                    continue
                n += 1
                fw2 = source == "fineweb2" or source.startswith("fw2:")
                yield text, f"{stem}:{doc_id}" if fw2 else str(doc_id)


def c4_lines(text: str) -> Iterator[str]:
    """The lines C4QualityFilter(split_paragraph=True) passes to split_into_sentences."""
    for line in text.splitlines():
        yield CITATION.sub("", line.strip())


def corpus_items(source: str, limit: int | None, lines: bool) -> Iterator[Item]:
    seen: set[str] = set()
    for text, doc_id in read_docs(source, limit):
        yield text, {"k": "doc", "id": doc_id}
        seen.add(text)
        if lines:
            for line in c4_lines(text):
                if line not in seen:
                    seen.add(line)
                    yield line, {"k": "line", "id": doc_id}


# --------------------------------------------------------------------------
# Fuzz pools (all sorted, so a seed always gives the same texts on one CPython)

UNIVERSE_RANGES = [(0, 0xD7FF), (0xE000, 0x3FFFF), (0xE0000, 0xE01EF)]
SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "latin1": [(0xC0, 0xFF)],
    "latinA": [(0x100, 0x17F)],
    "latinB": [(0x180, 0x24F)],
    "ipa": [(0x250, 0x2AF), (0x1D00, 0x1DBF)],
    "latin_add": [(0x1E00, 0x1EFF), (0x2C60, 0x2C7F), (0xA720, 0xA7FF), (0xAB30, 0xAB6F)],
    "greek": [(0x370, 0x3FF), (0x1F00, 0x1FFF)],
    "cyrillic": [(0x400, 0x52F), (0x1C80, 0x1C8F), (0xA640, 0xA69F)],
    "armenian": [(0x531, 0x58A)],
    "hebrew": [(0x591, 0x5F4), (0xFB1D, 0xFB4F)],
    "arabic": [(0x600, 0x6FF), (0x750, 0x77F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFC)],
    "devanagari": [(0x900, 0x97F)],
    "bengali": [(0x980, 0x9FF)],
    "tamil": [(0xB80, 0xBFF)],
    "telugu_kannada": [(0xC00, 0xCFF)],
    "sinhala": [(0xD80, 0xDFF)],
    "thai_lao": [(0xE00, 0xEFF)],
    "georgian": [(0x10A0, 0x10FF), (0x1C90, 0x1CBF)],
    "ethiopic": [(0x1200, 0x139F)],
    "cherokee": [(0x13A0, 0x13FD), (0xAB70, 0xABBF)],
    "hangul": [(0xAC00, 0xD7A3), (0x1100, 0x11FF)],
    "kana": [(0x3041, 0x30FF)],
    "cjk": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0xF900, 0xFAFF)],
    "fullwidth": [(0xFF10, 0xFF5A)],
    "math": [(0x1D400, 0x1D7FF)],
    "deseret_osage": [(0x10400, 0x1044F), (0x104B0, 0x104FB)],
    "adlam": [(0x1E900, 0x1E95F)],
}
# Blocks first assigned in Unicode 15.0 / 15.1 / 16.0: every code point is used,
# assigned or not, so the same texts hit version differences.
NEW_RANGES = [
    (0x11F00, 0x11F5F),  # Kawi (15.0)
    (0x1E4D0, 0x1E4FF),  # Nag Mundari (15.0)
    (0x1E030, 0x1E08F),  # Cyrillic Extended-D (15.0)
    (0x2FFC, 0x2FFF),  # Ideographic description characters (15.1)
    (0x31EF, 0x31EF),  # (15.1)
    (0x2EBF0, 0x2EE5F),  # CJK Extension I (15.1)
    (0x10D40, 0x10D8F),  # Garay (16.0)
    (0x16100, 0x1613F),  # Gurung Khema (16.0)
    (0x16D40, 0x16D7F),  # Kirat Rai (16.0)
    (0x1E5D0, 0x1E5FF),  # Ol Onal (16.0)
    (0x11BC0, 0x11BFF),  # Sunuwar (16.0)
    (0x105C0, 0x105FF),  # Todhri (16.0)
    (0x11380, 0x113FF),  # Tulu-Tigalari (16.0)
    (0x1CC00, 0x1CEBF),  # Symbols for Legacy Computing Supplement (16.0)
    (0x1FA80, 0x1FAFF),  # Symbols and Pictographs Extended-A (new emoji)
]
# fmt: off
ENGLISH = [
    "the", "a", "an", "I", "U", "A", "O", "and", "or", "but", "not", "no", "yes", "is", "are",
    "was", "were", "be", "been", "am", "do", "does", "did", "done", "can", "could", "will", "would",
    "shall", "should", "may", "might", "must", "have", "has", "had", "go", "going", "gonna",
    "gotta", "wanna", "lets", "let", "us", "we", "you", "he", "she", "it", "they", "them", "me",
    "him", "her", "its", "who", "what", "when", "where", "why", "how", "this", "that", "these",
    "those", "there", "here", "then", "than", "ill", "id", "wed", "well", "hell", "shell", "cant",
    "wont", "dont", "im", "ive", "youre", "whats", "st", "mr", "mrs", "dr", "jr", "sr", "co",
    "inc", "ltd", "vs", "etc", "eg", "ie", "pm", "jan", "feb", "mar", "apr", "jun", "aug", "sept",
    "oct", "nov", "dec", "mon", "tue", "hello", "world", "new", "york", "data", "file", "page",
    "time", "year", "people", "way", "day", "man", "thing", "woman", "life", "child", "email",
    "url", "http", "www", "com", "org", "net", "html", "quality", "filter", "token", "sentence",
    "word", "text", "cannot", "ain", "gimme", "lemme", "dunno", "nothin", "y", "all", "ma", "o",
    "clock",
]
EXTRA_WORDS = [
    "rock'n'roll", "o'clock", "O'Neil", "d'Artagnan", "l'homme", "'tis", "'em", "'cause", "'90s",
    "y'all", "ma'am", "ain't", "James'", "Jesus'", "students'", "it's", "Its", "she's", "he'S",
    "e.g.", "i.e.", "U.S.", "U.S.A.", "U.K.", "Ph.D.", "a.m.", "p.m.", "A.M.", "etc.", "vs.", "No.",
    "St.", "Mt.", "Mr.", "Mrs.", "Dr.", "Prof.", "Inc.", "Jan.", "Sept.", "approx.", "fig.", "Fig.",
    "Gov.", "Gen.", "Calif.", "N.Y.", "D.C.", "a.k.a.", "A.B.C.", "ABC.", "AB.", "Ab.", "aB.",
]
ASCII_LOWER = "abcdefghijklmnopqrstuvwxyz"
LABEL_CHARS = ASCII_LOWER + "0123456789"
PATH_CHARS = [*"abcxyz0123456789/?&=#%~()._-,'+!*;:@$[]", "%20", "é", "日", "…"]
LONG_CHARS = [*"abcdefgABCDEFG0123456789-.,:/'_+*^=<>", "…", "’"]
SUFFIX_EXTRA = ["'s", "'S", "’s", "’S", "—", "–", ".", "."]
APOSTROPHES = ["'", "’", "‘", "`", "´", "ʼ", "＇", "′", "ʹ"]
CONTRACTION_TAILS = ["s", "S", "t", "T", "d", "D", "ll", "LL", "re", "RE", "ve", "VE", "m", "M"]
HYPHEN_LIKE = ["-", "–", "—", "--", "---", "——", "~", "‐", "‑", "‒", "−", "⁃", "﹣", "－", "_"]
INFIX_OPS = [".", ",", ":", "<", ">", "=", "/", "+", "*", "^", "…", "...", "..", "&", "@", "|"]
OPENERS = [
    "(", "[", "{", "<", '"', "'", "“", "‘", "«", "„", "‚", "「", "『", "（", "【", "《", "〈",
]
CLOSERS = [
    ")", "]", "}", ">", '"', "'", "”", "’", "»", "“", "‘", "」", "』", "）", "】", "》", "〉",
]
TRAILERS = [
    ".", ",", "!", "?", ":", ";", "'s", "’s", "'S", "...", "…", ").", '."', ".”", "?!", "!!",
    "%", "+", "-", "—", "''", "'", "’", ")", "]", "''.", "s", "1", "²", ".,", ";)",
]
TLDS = [
    "com", "org", "net", "io", "co", "uk", "de", "info", "museum", "xn--p1ai", "рф", "中国",
    "onion", "x", "COM", "Com", "cOm", "co.uk", "com.au", "a1", "123", "ελ", "ⓒⓞⓜ", "é", "ab",
    "photography", "ac.jp", "مصر", "q" * 63, "q" * 64,
]
SCHEMES = [
    "http", "https", "ftp", "HTTP", "Https", "git+ssh", "svn+ssh", "file", "s3", "a", "x.y",
    "chrome-extension", "ws", "1http", "h-t", "mailto", "ht_tp", "é",
]
EMOJI = [
    "😀", "👍", "👍🏽", "❤️", "❤", "☺", "☺️", "🇺🇸", "🇬🇧", "👨‍👩‍👧", "🏳️‍🌈", "1️⃣", "#️⃣", "©", "®",
    "™", "🤷‍♀️", "🙏", "🔥", "✨", "💯", "🥲", "🫠", "🫨", "🩷",
    "🐦‍🔥", "🫩", "🪾", "🇦", "☝🏻", "✊🏿",
    "🏴\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f", "⚽", "✔", "✅", "⭐", "🎉",
]
SPECIAL_CPS = [
    "\x7f", "\x80", "\x9f", "\xad", "\U0000200b", "\U0000200c", "\U0000200d", "\U0000200e",
    "\U0000200f", "\U00002060", "\U00002061", "\U0000feff", "\U0000061c", "\U0000180e",
    "\U0000fff9", "\U0000fffc", "\U0000fffd", "\U0000fffe", "\U0000ffff", "\U000e0001",
    "\U000e0041", "\U0000e000", "\U0000f8ff", "\U000f0000", "\U0010fffd", "\U0010ffff",
    "\U00000378", "\U00002fff", "\U0001fffe", "\U0000034f", "\U000017b4", "\U00002064",
    "\U0000206f", "\U0001d173",
]
EDGE = [
    "", " ", "  ", "   ", "\t", "\n", "\n\n", "\r\n", "\r", "\xa0", "\U00002009", "\U0000202f",
    "\U00003000", "\x1c", "\x1d", "\x1e", "\x1f", "\x0b", "\x0c", "\x85", "\U00002028",
    "\U00002029", " \n ", "\t \t", " \xa0 ", "\n\xa0\n", "a", " a", "a ", " a ", "a  b", "a\tb",
    "a\nb", "a \n b", "a\xa0b", "a\U00003000b", ".", "..", "...", "....", "\U00002026", "a.",
    "A.", "AB.", "a. b", "Hello world.", "Hello world. This is it!", "!", "?", "?!",
    "\U00003002", "\U0000200b", "\U0000feff", "'", "''", '"', "C++", "'s", "a's", "\xb0F.",
    "5\xb0F.", "98.6\xb0F.", "\xb0C", ":)", "<3", "U.S.", "U.S.A.", "e.g.", "i.e.", "don't",
    "Don\U00002019t", "DON'T", "http://example.com", "www.example.com.", "user@example.com",
    "\\n", "\\t", "<space>", '\\")', "a.\n", "\n.", " .", ". ", "\n\n.\n\n",
]
# fmt: on


def _universe() -> Iterator[str]:
    for lo, hi in UNIVERSE_RANGES:
        for cp in range(lo, hi + 1):
            yield chr(cp)


def _unescape(s: str) -> str:
    return re.sub(r"\\(.)", r"\1", s)


@functools.cache
def pools() -> dict[str, Any]:
    """Character and string pools for the fuzzer, from spaCy's rules and Unicode."""
    import spacy
    from spacy.lang import char_classes as cc
    from spacy.lang.tokenizer_exceptions import emoticons
    from spacy.pipeline.sentencizer import Sentencizer

    by_cat: dict[str, list[str]] = {}
    for c in _universe():
        by_cat.setdefault(unicodedata.category(c), []).append(c)
    classes = {
        "alpha_lower": re.compile(f"[{cc.ALPHA_LOWER}]"),
        "alpha_upper": re.compile(f"[{cc.ALPHA_UPPER}]"),
        "alpha": re.compile(f"[{cc.ALPHA}]"),
        "icons": re.compile(f"[{cc.ICONS}]"),
    }
    members: dict[str, list[str]] = {k: [] for k in classes}
    for c in _universe():
        for k, pat in classes.items():
            if pat.match(c):
                members[k].append(c)
    alpha = set(members["alpha"])
    lower = set(members["alpha_lower"])
    letters = [c for cat in ("Lu", "Ll", "Lt", "Lm", "Lo") for c in by_cat.get(cat, [])]
    scripts: dict[str, list[str]] = {}
    for name, ranges in SCRIPT_RANGES.items():
        scripts[name] = [
            chr(cp)
            for lo, hi in ranges
            for cp in range(lo, hi + 1)
            if unicodedata.category(chr(cp))[0] in "LM"
        ]
    scripts["new"] = [chr(cp) for lo, hi in NEW_RANGES for cp in range(lo, hi + 1)]
    zeros = [c for c in by_cat["Nd"] if unicodedata.digit(c) == 0]
    nlp = spacy.blank("en")
    rules = sorted(k for k in nlp.tokenizer.rules if "IS_ALPHA" not in k)
    p: dict[str, Any] = {
        "rules": rules,
        "emoticons": sorted(emoticons),
        "sent": sorted(Sentencizer.default_punct_chars),
        "punct": [_unescape(x) for x in cc.LIST_PUNCT],
        "quotes": [_unescape(x) for x in cc.LIST_QUOTES],
        "currency": [_unescape(x) for x in cc.LIST_CURRENCY],
        "units": [_unescape(x) for x in cc.LIST_UNITS],
        "hyphens": [_unescape(x) for x in cc.LIST_HYPHENS],
        "ellipses": ["..", "...", "....", ".....", "…", "……", ". . .", "…."],
        "prefix_only": ["§", "%", "=", "—", "–", "+"],
        "icons": members["icons"],
        "lower": members["alpha_lower"],
        "upper": [c for c in members["alpha_upper"] if c not in lower],
        "alpha": members["alpha"],
        "other_letters": [c for c in letters if c not in alpha],
        "digit_sets": ["".join(chr(ord(z) + i) for i in range(10)) for z in zeros],
        "P": [c for cat in ("Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po") for c in by_cat[cat]],
        "S": [c for cat in ("Sm", "Sc", "Sk", "So") for c in by_cat[cat]],
        "Sc": by_cat["Sc"],
        "M": [c for cat in ("Mn", "Me", "Mc") for c in by_cat[cat]],
        "N_other": by_cat["No"] + by_cat["Nl"],
        "Z": by_cat["Zs"] + by_cat["Zl"] + by_cat["Zp"],
        "C": [c for c in by_cat["Cc"] if not c.isspace()] + by_cat["Cf"] + SPECIAL_CPS,
        "scripts": scripts,
        "script_names": sorted(scripts),
        "assigned": [
            c for cat, cs in sorted(by_cat.items()) if cat not in ("Cn", "Co", "Cs") for c in cs
        ],
        "universe": list(_universe()),
    }
    p["punct_run"] = (
        p["punct"] + p["quotes"] + p["sent"] + list(".,;:!?-'\"()[]{}<>/\\|@#$%^&*_+=~`")
    )
    p["mark_bases"] = [*p["punct"], ".", "-", "'"]
    return p


# fmt: off
JOINERS = [
    (" ", 60), ("  ", 5), ("\t", 3), ("\n", 6), ("\n\n", 3), ("\r\n", 3), ("\r", 1), ("\xa0", 3),
    ("\U00002009", 1), ("\U0000202f", 1), ("\U00003000", 2), ("\x1c", 1), ("\x1d", 0.5),
    ("\x1e", 0.5), ("\x1f", 0.5), ("\x0b", 0.5), ("\x0c", 0.5), ("\x85", 0.5),
    ("\U00002028", 0.5), ("\U00002029", 0.5), ("\U00001680", 0.3), ("\U00002000", 0.3),
    ("\U00002003", 0.3), ("\U0000200a", 0.3), ("\U0000205f", 0.3), (" \n", 2), ("\n ", 2),
    (" \t ", 1), ("   ", 1), (" \xa0", 1), ("\xa0 ", 1), ("\n\n\n", 0.5), (" \r\n", 0.5),
    ("\U0000200b", 0.5), ("\U0000feff", 0.3), ("\U00002060", 0.2), ("\U0000180e", 0.2), ("", 8),
]
# fmt: on


class Fuzzer:
    """Texts aimed at spaCy's English tokenizer rules and the sentencizer."""

    def __init__(self, seed: int, golden: bool = False) -> None:
        self.rng = random.Random(seed)
        self.golden = golden
        self.p = pools()
        self.joiners = [j for j, _ in JOINERS]
        self.joiner_cum = list(itertools.accumulate(w for _, w in JOINERS))
        self.pieces = self.piece_weights()
        self.texts = self.text_weights()
        self.piece_fns = [fn for fn, _ in self.pieces]
        self.piece_cum = list(itertools.accumulate(w for _, w in self.pieces))
        self.text_fns = [fn for fn, _ in self.texts]
        self.text_cum = list(itertools.accumulate(w for _, w in self.texts))

    def piece_weights(self) -> list[tuple[Callable[[], str], float]]:
        """The generators of one piece of text (a word, a number, a URL ...) and their weights."""
        return [
            (self.word, 25),
            (self.rule, 14),
            (self.number, 8),
            (self.url, 5),
            (self.email, 2),
            (self.contraction, 5),
            (self.hyphenated, 4),
            (self.infix, 6),
            (self.emoticon, 3),
            (self.sent_end, 8),
            (self.uni_punct, 5),
            (self.combining, 3),
            (self.emoji, 3),
            (self.cjk, 3),
            (self.control, 1.5),
            (self.random_cp, 2),
            (self.ellipsis, 2),
            (self.affixed, 8),
            (self.extra_word, 3),
            (self.punct_run, 3),
            (self.new_chars, 1),
        ]

    def text_weights(self) -> list[tuple[Callable[[], str], float]]:
        """The generators of a whole text and their weights."""
        return [
            (self.t_space, 4),
            (self.t_punct, 3),
            (self.t_single, 12),
            (self.t_glued, 15),
            (self.t_mixed, 34),
            (self.t_english, 20),
            (self.t_sentences, 8),
            (self.t_boundary, 10),
            (self.t_long, 0.15 if self.golden else 1.5),
            (self.t_soup, 0.7),
        ]

    # helpers
    def pick(self, seq: Sequence[str]) -> str:
        return seq[self.rng.randrange(len(seq))]

    def chance(self, p: float) -> bool:
        return self.rng.random() < p

    def joiner(self) -> str:
        return self.rng.choices(self.joiners, cum_weights=self.joiner_cum)[0]

    def piece(self) -> str:
        return self.rng.choices(self.piece_fns, cum_weights=self.piece_cum)[0]()

    def recase(self, s: str) -> str:
        r = self.rng.random()
        if r < 0.1:
            return s.upper()
        if r < 0.18:
            return s.title()
        if r < 0.22:
            return s.swapcase()
        if r < 0.25:
            return s.lower()
        return s

    def letters(self, script: str | None = None, n: int | None = None) -> str:
        if script is None:
            script = self.pick(self.p["script_names"])
        pool = self.p["scripts"][script]
        n = n or self.rng.randint(1, 3 if script in ("cjk", "kana", "hangul") else 8)
        return "".join(self.pick(pool) for _ in range(n))

    def digits(self, n: int | None = None) -> str:
        n = n or self.rng.randint(1, 5)
        ds = "0123456789" if self.chance(0.8) else self.pick(self.p["digit_sets"])
        if self.chance(0.05):  # mixed scripts
            return "".join(self.pick(self.pick(self.p["digit_sets"])) for _ in range(n))
        return "".join(self.pick(ds) for _ in range(n))

    # pieces
    def word(self) -> str:
        r = self.rng.random()
        if r < 0.45:
            return self.recase(self.pick(ENGLISH))
        if r < 0.65:
            return self.recase(
                "".join(self.pick(ASCII_LOWER) for _ in range(self.rng.randint(1, 10)))
            )
        if r < 0.72:
            return "".join(
                self.pick(self.p["lower"] if self.chance(0.7) else self.p["upper"])
                for _ in range(self.rng.randint(1, 6))
            )
        if r < 0.77:
            return "".join(
                self.pick(self.p["other_letters"]) for _ in range(self.rng.randint(1, 5))
            )
        return self.recase(self.letters())

    def extra_word(self) -> str:
        return self.recase(self.pick(EXTRA_WORDS))

    def rule(self) -> str:
        k = self.pick(self.p["rules"])
        r = self.rng.random()
        if r < 0.12:
            k = self.recase(k)
        elif r < 0.22:
            for a in ("'", "’"):
                if a in k:
                    k = k.replace(a, self.pick(APOSTROPHES))
        elif r < 0.26 and len(k) > 1:
            i = self.rng.randrange(len(k))
            k = k[:i] + k[i + 1 :] if self.chance(0.5) else k[:i] + k[i] + k[i:]
        return self.context(k)

    def context(self, k: str) -> str:
        r = self.rng.random()
        if r < 0.15:
            i = self.rng.randrange(len(OPENERS))
            k = OPENERS[i] + k + (CLOSERS[i] if self.chance(0.8) else self.pick(CLOSERS))
        elif r < 0.25:
            k = self.pick(OPENERS + self.p["prefix_only"] + self.p["currency"]) + k
        if self.chance(0.3):
            k += self.pick(TRAILERS)
        if self.chance(0.08):
            k = (self.word() if self.chance(0.6) else self.digits()) + k
        if self.chance(0.08):
            k += self.word() if self.chance(0.6) else self.digits()
        return k

    def number(self) -> str:
        r = self.rng.random()
        d = self.digits
        if r < 0.15:
            s = d()
        elif r < 0.3:
            s = d(self.rng.randint(1, 3)) + self.pick([",", ".", " ", "'", "٬", "،"]) + d(3)
        elif r < 0.4:
            s = d() + self.pick([".", ",", "٫", "·"]) + d()
        elif r < 0.5:
            s = d(self.rng.randint(1, 3)) + self.pick(self.p["units"])
        elif r < 0.58:
            c = self.pick(self.p["currency"] + self.p["Sc"][:20])
            s = c + d() if self.chance(0.5) else d() + c
        elif r < 0.64:
            s = d(2) + self.pick(["°F.", "°C", "°f.", "°K.", "° F", "°c.", "°", "°F", "℃", "°k"])
        elif r < 0.72:
            s = (
                d()
                + self.pick(["+", "-", "*", "^", "--", "/", ":", "x", "–", "+-", "-+"])
                + (d() if self.chance(0.8) else self.word())
            )
        elif r < 0.78:
            s = self.pick(["+", "-", "±", "−", "~", "#", "No.", "№"]) + d()
        elif r < 0.84:
            s = d(self.rng.randint(1, 2)) + self.pick(
                ["st", "nd", "rd", "th", "am", "pm", "a.m.", "p.m.", "AM", "%", "s", "'s", "k", "+"]
            )
        elif r < 0.9:
            s = self.pick(
                ["v", "V", "#", "0x", "x", "1e", "1E+", "1e-", "", "+1-", "(", "$"]
            ) + ".".join(d(self.rng.randint(1, 3)) for _ in range(self.rng.randint(2, 4)))
        elif r < 0.95:
            s = d(4) + self.pick(["-", "/", "."]) + d(2) + self.pick(["-", "/", "."]) + d(2)
        else:
            s = d() + self.pick(self.p["N_other"])
        return self.context(s) if self.chance(0.3) else s

    def url(self) -> str:
        rng = self.rng
        s = ""
        r = rng.random()
        if r < 0.55:
            s = self.pick(SCHEMES) + self.pick(["://", "://", "://", ":/", ":", ":///"])
        elif r < 0.7:
            s = self.pick(["www.", "WWW.", "ftp."])
        if self.chance(0.1):
            s += (
                self.pick(["user", "john.doe", "a", "u-1", "é", "x"])
                + (":" + self.pick(["", "pass", "p@ss", "1", "%40"]) if self.chance(0.5) else "")
                + "@"
            )
        if self.chance(0.15):
            s += self.ip()
        else:
            labels = []
            for _ in range(rng.randint(1, 3)):
                q = rng.random()
                if q < 0.6:
                    lab = "".join(self.pick(LABEL_CHARS) for _ in range(rng.randint(1, 12)))
                elif q < 0.7:
                    lab = self.recase(self.pick(["example", "my-site", "a_b", "x-", "-x", "_x"]))
                elif q < 0.85:
                    lab = self.letters(self.pick(["latin1", "cyrillic", "cjk", "greek", "arabic"]))
                elif q < 0.9:
                    lab = self.pick(EMOJI)
                else:
                    lab = self.recase(self.word())
                labels.append(lab)
            s += ".".join(labels) + "." + self.pick(TLDS)
        if self.chance(0.15):
            s += ":" + self.digits(rng.randint(1, 6))
        if self.chance(0.5):
            s += self.pick(["/", "?", "#", "/", "/"]) + "".join(
                self.pick(PATH_CHARS) for _ in range(rng.randint(0, 25))
            )
        if self.chance(0.3):
            s += self.pick(TRAILERS)
        if self.chance(0.15):
            i = rng.randrange(len(OPENERS))
            s = OPENERS[i] + s + CLOSERS[i]
        return s

    def ip(self) -> str:
        rng = self.rng

        def octet() -> str:
            r = rng.random()
            if r < 0.3:
                return str(rng.choice([0, 1, 9, 10, 99, 100, 127, 168, 169, 172, 192, 199, 200]))
            if r < 0.5:
                return str(rng.choice([223, 224, 249, 250, 254, 255, 256, 299, 999]))
            if r < 0.55:
                return "0" + str(rng.randint(0, 99))
            return str(rng.randint(0, 255))

        r = rng.random()
        if r < 0.3:
            head = self.pick(
                ["10", "127", "169.254", "192.168", "172.16", "172.31", "172.15", "172.32"]
            )
            n = 4 - head.count(".") - 1
        else:
            head = octet()
            n = 3
        n += rng.choice([0, 0, 0, -1, 1])
        return ".".join([head, *(octet() for _ in range(max(n, 0)))])

    def email(self) -> str:
        local = "".join(self.pick("abcxyz019._%+-") for _ in range(self.rng.randint(1, 10)))
        dom = "".join(self.pick("abcxyz019-") for _ in range(self.rng.randint(1, 10)))
        s = local + "@" + dom + "." + self.pick(TLDS)
        if self.chance(0.15):
            s = "mailto:" + s
        return self.context(s) if self.chance(0.3) else s

    def contraction(self) -> str:
        r = self.rng.random()
        a = self.pick(APOSTROPHES)
        if r < 0.5:
            s = self.word() + a + self.pick(CONTRACTION_TAILS)
        elif r < 0.7:
            s = a + self.word()
        elif r < 0.85:
            s = self.word() + a
        else:
            s = self.word() + a + self.word() + (a + self.word() if self.chance(0.3) else "")
        return self.context(s) if self.chance(0.3) else s

    def hyphenated(self) -> str:
        parts = [self.word() for _ in range(self.rng.randint(2, 4))]
        h = self.pick(self.p["hyphens"] + HYPHEN_LIKE)
        if self.chance(0.7):
            return h.join(parts)
        return "".join(p + self.pick(HYPHEN_LIKE) for p in parts[:-1]) + parts[-1]

    def operand(self) -> str:
        r = self.rng.random()
        if r < 0.3:
            return self.pick(self.p["lower"])
        if r < 0.5:
            return self.pick(self.p["upper"])
        if r < 0.65:
            return self.digits(1)
        if r < 0.75:
            return self.pick(self.p["quotes"])
        if r < 0.85:
            return self.word()
        return self.pick(self.p["other_letters"] + self.p["M"][:50])

    def infix(self) -> str:
        n = self.rng.randint(2, 5)
        out = [self.operand()]
        for _ in range(n - 1):
            out.append(self.pick(INFIX_OPS + self.p["hyphens"] + self.p["icons"][:40]))
            out.append(self.operand())
        return "".join(out)

    def emoticon(self) -> str:
        return self.context(self.pick(self.p["emoticons"]))

    def sent_end(self) -> str:
        r = self.rng.random()
        if r < 0.5:
            p = self.pick([".", "!", "?"])
        elif r < 0.7:
            p = "".join(self.pick([".", "!", "?"]) for _ in range(self.rng.randint(2, 4)))
        elif r < 0.9:
            p = self.pick(self.p["sent"])
        else:
            p = "".join(self.pick(self.p["sent"]) for _ in range(self.rng.randint(2, 3)))
        if self.chance(0.3):
            p += self.pick(CLOSERS) + (self.pick(CLOSERS) if self.chance(0.2) else "")
        sep = " " if self.chance(0.1) else ""
        return self.word() + sep + p

    def uni_punct(self) -> str:
        s = "".join(
            self.pick(self.p["P"] if self.chance(0.6) else self.p["S"])
            for _ in range(self.rng.randint(1, 3))
        )
        r = self.rng.random()
        if r < 0.3:
            return self.word() + s
        if r < 0.5:
            return s + self.word()
        if r < 0.6:
            return self.word() + s + self.word()
        return s

    def combining(self) -> str:
        marks = "".join(self.pick(self.p["M"]) for _ in range(self.rng.randint(1, 3)))
        r = self.rng.random()
        if r < 0.5:
            w = self.word()
            i = self.rng.randint(1, len(w))
            return w[:i] + marks + w[i:]
        if r < 0.7:
            return marks
        if r < 0.85:
            return self.pick(self.p["mark_bases"]) + marks
        return self.digits(1) + marks

    def emoji(self) -> str:
        s = self.pick(EMOJI) if self.chance(0.7) else self.pick(self.p["icons"])
        if self.chance(0.3):
            s = s * self.rng.randint(2, 4)
        if self.chance(0.3):
            s = self.word() + s if self.chance(0.5) else s + self.word()
        return s

    def cjk(self) -> str:
        s = self.letters(self.pick(["cjk", "kana", "hangul"]), self.rng.randint(1, 10))
        if self.chance(0.5):
            s += self.pick(
                ["。", "、", "「", "」", "，", "！", "？", "：", "；", "・", "ー", "〜", "）"]
            )
        return s

    def control(self) -> str:
        c = "".join(self.pick(self.p["C"]) for _ in range(self.rng.randint(1, 2)))
        return (
            self.word() + c + (self.word() if self.chance(0.5) else "") if self.chance(0.6) else c
        )

    def random_cp(self) -> str:
        pool = self.p["universe"] if self.chance(0.3) else self.p["assigned"]
        return "".join(self.pick(pool) for _ in range(self.rng.randint(1, 5)))

    def new_chars(self) -> str:
        s = "".join(self.pick(self.p["scripts"]["new"]) for _ in range(self.rng.randint(1, 4)))
        return self.context(s) if self.chance(0.4) else s

    def ellipsis(self) -> str:
        e = self.pick(self.p["ellipses"])
        r = self.rng.random()
        if r < 0.3:
            return self.word() + e + self.word()
        if r < 0.6:
            return self.word() + e
        if r < 0.8:
            return e + self.word()
        return e

    def affixed(self) -> str:
        pre = "".join(
            self.pick(
                self.p["prefix_only"]
                + self.p["punct"]
                + self.p["quotes"]
                + self.p["currency"]
                + self.p["ellipses"]
                + self.p["icons"][:30]
            )
            for _ in range(self.rng.choice([0, 1, 1, 1, 2, 3]))
        )
        suf = "".join(
            self.pick(
                self.p["punct"]
                + self.p["quotes"]
                + self.p["ellipses"]
                + self.p["icons"][:30]
                + SUFFIX_EXTRA
            )
            for _ in range(self.rng.choice([0, 1, 1, 1, 2, 3]))
        )
        core = self.word() if self.chance(0.7) else self.number()
        return pre + core + suf

    def punct_run(self) -> str:
        pool = self.p["punct_run"]
        return "".join(self.pick(pool) for _ in range(self.rng.randint(1, 6)))

    # whole texts
    def target_len(self) -> int:
        r = self.rng.random()
        if self.golden:  # keep the committed file small
            if r < 0.6:
                return self.rng.randint(1, 12)
            if r < 0.95:
                return self.rng.randint(12, 35)
            return self.rng.randint(35, 80)
        if r < 0.3:
            return self.rng.randint(1, 30)
        if r < 0.75:
            return self.rng.randint(30, 150)
        return self.rng.randint(150, 400)

    def join(self, pieces: list[str]) -> str:
        out = []
        for i, p in enumerate(pieces):
            if i:
                out.append(self.joiner())
            out.append(p)
        s = "".join(out)
        if self.chance(0.1):
            s = self.joiner() + s
        if self.chance(0.1):
            s += self.joiner()
        return s

    def t_space(self) -> str:
        return "".join(self.joiner() for _ in range(self.rng.randint(0, 4)))

    def t_punct(self) -> str:
        return self.join([self.punct_run() for _ in range(self.rng.randint(1, 6))])

    def t_single(self) -> str:
        return self.piece()

    def t_glued(self) -> str:
        return "".join(self.piece() for _ in range(self.rng.randint(2, 2 if self.golden else 6)))

    def english_word(self) -> str:
        r = self.rng.random()
        if r < 0.75:
            return self.pick(ENGLISH)
        if r < 0.85:  # a lower-case rule: contraction, abbreviation, emoticon ...
            return self.pick(self.p["rules"])
        if r < 0.9:
            return self.pick(EXTRA_WORDS)
        return self.piece()

    def t_english(self) -> str:
        """Prose-like text: capitalised sentences, commas, quotes, numbers, links."""
        n = self.target_len()
        out: list[str] = []
        total = 0
        while total < n:
            n_words = self.rng.randint(1, 5 if self.golden else 12)
            words = [self.english_word() for _ in range(n_words)]
            words[0] = words[0][:1].upper() + words[0][1:]
            for i in range(1, len(words)):
                r = self.rng.random()
                if r < 0.08:
                    words[i - 1] += ","
                elif r < 0.11:
                    words[i] = self.pick(['"', "“", "(", "'", "‘"]) + words[i]
                elif r < 0.14:
                    words[i] += self.pick(['"', "”", ")", "'", "’", "'s", "’s"])
                elif r < 0.17:
                    words[i] = self.number()
                elif r < 0.19:
                    words[i] = self.url() if self.chance(0.7) else self.email()
                elif r < 0.21:
                    words[i] = self.hyphenated()
            end = self.pick([".", ".", ".", "!", "?", "...", "…", '."', ".”", "?!", ":", ""])
            s = " ".join(words) + end
            out.append(s)
            total += len(s) + 1
        return "".join(s + self.pick([" ", " ", " ", " ", "\n", "  ", "\n\n", ""]) for s in out)

    def t_mixed(self) -> str:
        n = self.target_len()
        pieces: list[str] = []
        total = 0
        while total < n:
            p = self.piece()
            pieces.append(p)
            total += len(p) + 1
        return self.join(pieces)

    def t_sentences(self) -> str:
        out = []
        for _ in range(self.rng.randint(1, 2 if self.golden else 6)):
            words = [
                self.word() if self.chance(0.8) else self.piece()
                for _ in range(self.rng.randint(0, 4 if self.golden else 7))
            ]
            out.append(" ".join([*words, self.sent_end()]))
        return "".join(o + self.pick([" ", " ", " ", "\n", "  ", "", "\n\n", "\xa0"]) for o in out)

    def t_boundary(self) -> str:
        pool = [*self.p["sent"], ".", ".", "!", "?", "...", "…", "?!", '."', ".”", ".)", ".'", "!”"]
        out = []
        for _ in range(self.rng.randint(1, 5 if self.golden else 8)):
            r = self.rng.random()
            if r < 0.4:
                out.append(self.word())
            elif r < 0.75:
                out.append(self.pick(pool))
            elif r < 0.85:
                out.append(self.pick(CLOSERS + OPENERS))
            else:
                out.append(self.piece())
        return self.join(out)

    def t_long(self) -> str:
        rng = self.rng
        n = rng.randint(1000, 1200) if self.golden else rng.randint(1000, 5000)
        r = rng.random()
        if r < 0.2:
            c = self.pick(
                ["a", "x", "Z", ".", "-", "!", "'", "é", "日", "😀", "_", "0", "/", "…", ":"]
            )
            s = c * n
        elif r < 0.35:
            s = "".join(
                self.pick(self.p["scripts"][self.pick(["latin1", "cyrillic", "greek"])])
                if rng.random() < 0.1
                else self.pick("abcdefghij")
                for _ in range(n)
            )
        elif r < 0.55:
            s = "".join(self.pick(LONG_CHARS) for _ in range(n))
        elif r < 0.65:
            s = "http://example.com/" + "".join(self.pick("abc/._-?=&%") for _ in range(n))
        elif r < 0.85:
            unit = self.pick(
                [
                    "a.",
                    "a-",
                    "1+",
                    "A.",
                    "a.B",
                    "ab,",
                    "x:y",
                    "'a",
                    "a'",
                    "(a)",
                    "Ab-",
                    "1.",
                    "--a",
                    ":)",
                    "U.S.",
                ]
            )
            s = unit * (n // len(unit))
        else:
            s = "".join(self.pick(self.p["P"]) for _ in range(n))
        if rng.random() < 0.3:
            s = self.word() + " " + s
        if rng.random() < 0.3:
            s += self.pick(TRAILERS) + (" " + self.word() if rng.random() < 0.5 else "")
        return s

    def t_soup(self) -> str:
        return "".join(
            self.pick(self.p["universe"])
            for _ in range(self.rng.randint(1, 20 if self.golden else 50))
        )

    def text(self) -> tuple[str, str]:
        fn = self.rng.choices(self.text_fns, cum_weights=self.text_cum)[0]
        while True:
            s = fn()
            if "IS_ALPHA" not in s:
                return s, fn.__name__[2:]


def fuzz_items(n: int, seed: int, golden: bool = False) -> Iterator[Item]:
    f = make_fuzzer(LANG, seed, golden)
    count = 0
    for s in EDGE:
        if count == n:
            return
        count += 1
        yield s, {"k": "edge"}
    while count < n:
        s, kind = f.text()
        if any(0xD800 <= ord(c) <= 0xDFFF for c in s):
            raise AssertionError("lone surrogate generated")
        count += 1
        yield s, {"k": kind}


def unicode_items(n: int, seed: int) -> Iterator[Item]:
    """Texts aimed at Unicode-version differences, identical on every CPython.

    Code points from blocks new in Unicode 15.0-16.0 (assigned or not) go where
    the version matters: after a sentence end (the sentencizer's `is_punct`),
    in URL schemes and hosts (`\\w`), in IP octets (`\\d`), glued to words.
    Only constants feed the generator, never `unicodedata`.
    """
    rng = random.Random(seed)
    new = [chr(cp) for lo, hi in NEW_RANGES for cp in range(lo, hi + 1)]
    enders = [".", "!", "?", "...", "…", "。", "।", "?!", '."']

    def run() -> str:
        return "".join(rng.choice(new) for _ in range(rng.randint(1, 3)))

    def word() -> str:
        return rng.choice(ENGLISH)

    shapes: list[Callable[[], str]] = [
        lambda: word() + rng.choice(enders) + rng.choice(["", " ", "\n"]) + run(),
        lambda: word() + rng.choice(enders) + " " + run() + rng.choice(enders) + " " + run(),
        lambda: word() + rng.choice(enders) + " " + run() + " " + word(),
        lambda: word() + " " + run() + rng.choice(enders),
        lambda: run() + rng.choice(["", " ", "."]) + word(),
        lambda: word() + run() + word(),
        lambda: run() + "://example.com/" + word(),
        lambda: "http://" + run() + "." + rng.choice(["com", "org", run()]),
        lambda: "http://" + word() + ".com/" + run() + rng.choice(["", ".", ")"]),
        lambda: ".".join(rng.choice(["8", "77", "123", run()]) for _ in range(4)),
        lambda: word() + "-" + run() + rng.choice(["", "-" + word()]),
        lambda: rng.choice(['"', "(", "“"]) + run() + rng.choice(['"', ")", "”"]),
    ]
    for _ in range(n):
        k = rng.randrange(len(shapes))
        parts = [shapes[k]() for _ in range(rng.randint(1, 3))]
        yield " ".join(parts), {"k": f"unicode{k}"}


# --------------------------------------------------------------------------
# Golden: diverse short real texts plus a fixed-seed fuzz run

FLAGS = {
    "url": re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE),
    "email": re.compile(r"\S+@\S+\.\w+"),
    "ip": re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    "citation": CITATION,
    "apos": re.compile(r"\w'\w"),
    "curly_apos": re.compile(r"\w’\w"),
    "possessive": re.compile(r"\w['’]s\b"),
    "ellipsis": re.compile(r"\.\.\.|…"),
    "unit": re.compile(r"\d+(?:km|kg|cm|mm|m|g|mg|lb|oz|ft|in|mph|GB|MB|KB|TB|%)\b"),
    "currency_num": re.compile(r"[$£€¥]\d"),
    "num_sep": re.compile(r"\d[,.]\d"),
    "range": re.compile(r"\d[-–—]\d"),
    "time": re.compile(r"\d(?:am|pm|a\.m\.|p\.m\.)", re.IGNORECASE),
    "degree": re.compile(r"°"),
    "abbrev": re.compile(r"\b(?:[A-Z]\.){2,}|\b(?:Mr|Mrs|Dr|St|Jr|Inc|Ltd|vs|etc|e\.g|i\.e)\."),
    "allcaps": re.compile(r"\b[A-Z]{3,}\b"),
    "hyphen_word": re.compile(r"[A-Za-z]-[A-Za-z]"),
    "double_hyphen": re.compile(r"--"),
    "emdash_glued": re.compile(r"\w[—–]\w"),
    "slash_word": re.compile(r"[A-Za-z]/[A-Za-z]"),
    "colon_word": re.compile(r"[A-Za-z0-9]:[A-Za-z]"),
    "dot_upper": re.compile(r"[a-z]\.[A-Z]"),
    "comma_glued": re.compile(r"[A-Za-z],[A-Za-z]"),
    "quote_after_period": re.compile(r"[.!?][\"'”’)]"),
    "multi_punct": re.compile(r"[!?]{2,}"),
    "emoticon": re.compile(r"(?<!\S)(?::\)|:-\)|:\(|;\)|:D|:P|<3|\^_\^|:p|xD)(?!\S)"),
    "crlf": re.compile(r"\r\n"),
    "tab": re.compile(r"\t"),
    "multispace": re.compile(r"  "),
    "blank_line": re.compile(r"\n\s*\n"),
    "line_punct_start": re.compile(r"(?m)^[^\w\s]"),
    "plus_word": re.compile(r"\+[A-Za-z]"),
    "hash": re.compile(r"#\w"),
    "at_mention": re.compile(r"(?<!\S)@\w"),
    "html_entity": re.compile(r"&#?\w+;"),
    "cpp": re.compile(r"C\+\+"),
}


def features(text: str) -> frozenset[str]:
    f = set()
    for c in set(text):
        if c.isascii():
            if not c.isalnum():
                f.add("c:" + c)
            continue
        cat = unicodedata.category(c)
        if cat[0] in "PSZCN" or cat == "Lm":
            f.add("c:" + c)
        else:
            f.add(f"k:{cat}:{unicodedata.name(c, '?').split(' ')[0]}")
    f.update("f:" + k for k, pat in FLAGS.items() if pat.search(text))
    return frozenset(f)


def windows(text: str, size: int = 180, n: int = 6) -> list[str]:
    """`text` itself if short, else up to `n` excerpts starting at line starts."""
    if len(text) <= 300:
        return [text]
    starts = [0, *(m.end() for m in re.finditer("\n", text) if m.end() < len(text) - 50)]
    picks = sorted({starts[i * (len(starts) - 1) // max(n - 1, 1)] for i in range(n)})
    out = []
    for s in picks:
        w = text[s : s + size]
        cut = max(w.rfind(" "), w.rfind("\n"))
        out.append(w[:cut] if cut > size // 2 else w)
    return out


def greedy(
    cands: list[tuple[str, frozenset[str]]], k: int, covered: set[str], scale: int = 300
) -> list[int]:
    """Lazy greedy set cover: up to `k` candidates adding the most new features per length."""

    def score(i: int) -> float:
        text, feats = cands[i]
        return len(feats - covered) / (1 + len(text) / scale)

    heap = [(-score(i), i) for i in range(len(cands))]
    heapq.heapify(heap)
    chosen: list[int] = []
    while heap and len(chosen) < k:
        _, i = heapq.heappop(heap)
        s = score(i)
        if s <= 0:
            continue
        if heap and s < -heap[0][0]:
            heapq.heappush(heap, (-s, i))
            continue
        chosen.append(i)
        covered |= cands[i][1]
    return chosen


def golden_real_items(n_docs: int = 140, n_lines: int = 80) -> list[Item]:
    import pyarrow.parquet as pq

    adv = pq.read_table(
        SOURCES["adversarial"][0][0], columns=["id", "text", "kind"], use_threads=False
    ).to_pylist()
    dev = list(read_docs("dev"))
    covered: set[str] = set()
    # Every adversarial edge case (bar the two multi-kB runs of one letter) ...
    chosen = [r["text"] for r in adv if r["kind"] == "edge" and len(r["text"]) <= 1000]
    # ... one excerpt per other adversarial kind ...
    by_kind: dict[str, list[str]] = {}
    for r in adv:
        if r["kind"] != "edge":
            by_kind.setdefault(r["kind"], []).append(r["text"])
    for kind in sorted(by_kind):
        cands = [(w, features(w)) for t in by_kind[kind][:20] for w in windows(t)]
        chosen.extend(cands[i][0] for i in greedy(cands, 1, covered))
    for t in chosen:
        covered |= features(t)
    # ... then the most diverse excerpts of everything.
    pool = [(w, features(w)) for r in adv for w in windows(r["text"])]
    pool += [(w, features(w)) for t, _ in dev for w in windows(t)]
    seen = set(chosen)
    picked = [
        pool[i][0] for i in greedy(pool, n_docs - len(chosen), covered) if pool[i][0] not in seen
    ]
    chosen.extend(picked)
    items: list[Item] = []
    seen_lines: set[str] = set(chosen)
    for t in chosen:
        items.append((t, {"k": "excerpt"}))
        for line in c4_lines(t):
            if line not in seen_lines:
                seen_lines.add(line)
                items.append((line, {"k": "line"}))
    # Extra C4-style lines from the whole dev and adversarial sets.
    line_pool: list[tuple[str, frozenset[str]]] = []
    for t in itertools.chain((r["text"] for r in adv), (t for t, _ in dev)):
        for line in c4_lines(t):
            if 0 < len(line) <= 150 and line not in seen_lines:
                seen_lines.add(line)
                line_pool.append((line, features(line)))
    items.extend((line_pool[i][0], {"k": "line"}) for i in greedy(line_pool, n_lines, covered))
    return items


# --------------------------------------------------------------------------
# Other languages: datatrove's assignment, per-language pools and fuzzers

# The datatrove language code each multilingual golden-KEY.jsonl is recorded with:
# the tokenizer's own language (its code_3_script; the FineWeb-2 subset where
# datatrove has several), else the biggest FineWeb-2 subset assigned to it.
# fmt: off
REPRESENTATIVE: dict[str, str] = {
    "af": "afr_Latn", "am": "amh_Ethi", "ar": "arb_Arab", "az": "azj_Latn", "bg": "bul_Cyrl",
    "ca": "cat_Latn", "cs": "ces_Latn", "da": "dan_Latn", "de": "deu_Latn", "dsb": "dsb_Latn",
    "el": "ell_Grek", "en": "eng_Latn", "es": "spa_Latn", "et": "ekk_Latn", "eu": "eus_Latn",
    "fa": "fas_Arab", "fi": "fin_Latn", "fo": "fao_Latn", "fr": "fra_Latn", "ga": "gle_Latn",
    "grc": "grc_Grek", "he": "heb_Hebr", "hr": "hrv_Latn", "hsb": "hsb_Latn", "hu": "hun_Latn",
    "hy": "hye_Armn", "id": "ind_Latn", "is": "isl_Latn", "it": "ita_Latn", "ky": "kir_Cyrl",
    "la": "lat_Latn", "lb": "ltz_Latn", "lg": "lug_Latn", "lij": "lij_Latn", "lt": "lit_Latn",
    "lv": "lvs_Latn", "mk": "mkd_Cyrl", "ms": "zsm_Latn", "nb": "nob_Latn", "nl": "nld_Latn",
    "pl": "pol_Latn", "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl", "sk": "slk_Latn",
    "sl": "slv_Latn", "sq": "als_Latn", "sr": "srp_Cyrl", "sv": "swe_Latn", "ti": "tir_Ethi",
    "tl": "fil_Latn", "tn": "tsn_Latn", "tr": "tur_Latn", "tt": "tat_Cyrl", "uk": "ukr_Cyrl",
    "xx": "uzn_Latn", "yo": "yor_Latn",
    # other tokenizer classes
    "ws": "kat_Geor", "indic-hi": "hin_Deva", "indic-ur": "urd_Arab", "indic-bn": "ben_Beng",
}
# fmt: on
GOLDEN_LANG_FUZZ = 900  # at most this many fuzz rows in a multilingual golden ...
GOLDEN_LANG_BYTES = 60_000  # ... and at most this many bytes of them
GOLDEN_LANG_CANDIDATES = 4  # fuzz texts generated (and recorded) per row kept

# fmt: off
# Hand-picked strings per golden key, on top of the tokenizer's own rules.
LANG_EXTRAS: dict[str, list[str]] = {
    "fr": [
        "l'homme", "L'HOMME", "l’homme", "L’Homme", "aujourd'hui", "Aujourd’hui", "AUJOURD'HUI",
        "peut-être", "PEUT-ÊTRE", "Peut‐être", "c'est-à-dire", "C’EST-À-DIRE", "qu'il", "jusqu'à",
        "d'accord", "est-ce", "a-t-il", "va-t'en", "rendez-vous", "porte-monnaie", "Saint-Étienne",
        "c.-à-d.", "M.", "Mme", "Mlle", "etc.", "presqu'île", "entr'ouvert", "grand'mère",
        "prud'homme", "week-end", "New-York", "États-Unis", "hip-hop", "1er", "2e", "XIXe", "n°",
        "«\xa0Bonjour\xa0»", "« oui »", "lorsqu'", "puisqu’", "quelqu'un", "s'il", "S'IL",
        "Bonjour\xa0!", "Quoi\u202f?", "anti-inflammatoire", "Jean-Pierre", "vis-à-vis",
        "arc-en-ciel", "pomme-de-terre", "sous-sol", "contre-attaque", "eux-mêmes", "dis-moi",
        "donne-le-moi", "y a-t-il", "t’es", "j'suis", "aujourd", "d’", "D'",
    ],
    "hu": [
        "2020-ban", "3.-án", "10%-kal", "1-jén", "2012.", "III.", "5:30-kor", "12,5%-os", "1990-es",
        "€-ért", "3-as", "A4-es", "100-200", "1/2-es", "2+2=4-et", "°C-on", "km/h", "stb.", "pl.",
        "dr.", "Bp.", "ún.", "ill.", "kb.", "1.000-ből", "XX.", "2.-ig", "1-2.", "%-os",
        "3×4-es", "1990-es", "10:15:30-kor", "5.5-ös", "$-t", "ő", "Ő", "ű", "Ű",
    ],
    "tr": [
        "1990'da", "TBMM'de", "Dr.'a", "3.'sü", "15:30'da", "1.000'den", "IV.", "ABD'ye",
        "2023'te", "Türkiye'nin", "İstanbul'da", "İSTANBUL", "ıIİi", "Prof.Dr.", "vb.", "vs.",
        "örn.", "1990’larda", "%50'si", "10/3'ü", "XX.", "Av.'ın", "No.'lu", "1.'nci",
        "12.05.2020'de", "3/4'ü", "Ank.'da", "İ", "ı", "Iİ", "AB'nin", "ŞÜ",
    ],
    "de": [
        "z.B.", "z. B.", "d.h.", "u.a.", "usw.", "bzw.", "Str.", "Nr.", "geht's", "gibt's", "wie's",
        "Haus-", "Bundes-", "(Haus-)Tür", "20.", "1.000,50", "3,5%", "§ 12", "Abs.", "Dr.", "Prof.",
        "Mio.", "Mrd.", "ca.", "„Hallo“", "»Ja«", "S-Bahn", "E-Mail", "1990er", "U-Bahn", "ß",
        "STRAßE", "Straße", "ẞ", "o.ä.", "u.s.w.", "evtl.", "inkl.", "zzgl.", "Jh.", "v.Chr.",
    ],
    "es": [
        "EE.UU.", "EE. UU.", "Sr.", "Sra.", "Dña.", "1.º", "1.ª", "n.º", "dámelo", "¿Qué?",
        "¡Hola!", "del", "al", "pág.", "etc.", "Ud.", "Uds.", "1.000,5", "«sí»", "año", "AÑO",
        "aprox.", "a.C.", "d.C.", "¿¡", "Ñ", "1º", "2ª",
    ],
    "ru": [
        "т.е.", "т. е.", "т.д.", "и т.д.", "и т. д.", "г.", "гг.", "ул.", "д.", "руб.", "млн", "ё",
        "Ё", "по-русски", "кое-что", "1 000", "XX в.", "т.к.", "т. к.", "см.", "др.", "«Да»", "—",
        "в т.ч.", "и др.", "н.э.", "до н. э.", "1-й", "2-го", "90-х", "№ 5", "„да“",
    ],
    "ar": [
        "ق.م", "ص.ب", "د.", "وال", "٫", "٬", "،", "؛", "؟", "ـ", "ﷺ", "١٢٣", "٣٫١٤", "١٬٠٠٠",
        "أ.د.", "م.", "هـ", "ب. م", "«نعم»", "إلخ", "ـــ", "وَ", "الْ", "ﻻ", "لا",
    ],
    "fa": ["می\u200cشود", "نمی\u200c", "۱۲۳", "۳٫۱۴", "ق.م", "؟", "،", "«بله»", "\u200c", "ها"],
    "it": ["l'uomo", "dell'anno", "all’inizio", "po'", "c'è", "un'altra", "qual è", "S.p.A."],
    "pt": ["d'água", "copo-d'água", "dá-lo", "fá-lo-ia", "Sr.ª", "n.º", "pág.", "etc."],
    "ca": ["l'home", "d'acord", "dóna-m'ho", "col·legi", "COL·LEGI", "l·l", "porta'l", "n'hi"],
    "nl": ["'s-Hertogenbosch", "'t", "z'n", "m'n", "d.w.z.", "o.a.", "zo'n", "auto's"],
    "el": ["κ.λπ.", "π.χ.", "ό,τι", "Ά", "ΐ", "ς", ";", "\u037e", "·", "\u0387", "Ο.Η.Ε."],
    "uk": ["п'ять", "м’ясо", "т.д.", "р.", "грн", "ґ", "Ї", "і"],
    "he": ["ש\"ח", "צה\"ל", "ארה״ב", "בע\"מ", "׳", "״"],
    "hy": ["։", "՝", "՞", "՜", "և", "Եվ"],
    "am": ["።", "፣", "፤", "፥", "፦", "፧", "፨", "፩፪"],
    "ti": ["።", "፣", "፤", "፧", "፩"],
    "id": ["anak-anak", "di-", "ke-2", "Rp10.000", "Rp 10.000,00", "dll.", "dsb."],
    "ms": ["kanak-kanak", "RM10", "dll.", "dsb.", "Kor."],
    "sr": ["да л'", "да л’", "Ђ", "ђ", "Љ", "љ", "dž", "Dž"],
    "pl": ["itd.", "np.", "tzw.", "ul.", "ł", "Ł", "ż"],
    "cs": ["atd.", "např.", "tzv.", "ř", "Ř", "ů"],
    "sv": ["t.ex.", "bl.a.", "osv.", "å", "Å"],
    "da": ["bl.a.", "f.eks.", "osv.", "ø", "Ø", "æ"],
    "nb": ["bl.a.", "f.eks.", "osv.", "ø", "Ø", "å"],
    "fi": ["esim.", "ns.", "jne.", "EU:n", "1990-luvulla", "ä"],
    "ro": ["ș", "ț", "Ș", "ş", "ţ", "într-o", "dintr-un", "s-a"],
    "lb": ["d'Kanner", "z.B.", "ë", "é"],
    "ga": ["n-athair", "t-uisce", "h-oíche", "mBaile", "Á"],
    "lij": ["l'é", "ch'o", "çittæ", "æ"],
    "ws": ["ა.შ.", "ე.ი.", "ქ.", "მ.შ.", "Ა.Ბ.", "Ⴀ.", "ⴀ"],
}
# Common words where no FineWeb-2 slice or spaCy stop list provides them.
LANG_WORDS: dict[str, list[str]] = {
    "ws": [
        "და", "არის", "ეს", "რომ", "არ", "მე", "ის", "საქართველო", "თბილისი", "ქართული", "ენა",
        "წელი", "როგორც", "უნდა", "კი", "ან", "მაგრამ", "ძალიან", "კარგი", "დღეს", "ახალი", "ყველა",
        "ერთი", "ორი", "სამი", "შემდეგ", "ასევე", "იყო", "ჩვენ", "თქვენ", "ᲡᲐᲥᲐᲠᲗᲕᲔᲚᲝ",
    ],
    "indic-hi": [
        "है", "और", "के", "में", "की", "को", "से", "यह", "भारत", "हिंदी", "नहीं", "था", "हैं", "पर",
        "भी", "श्री", "डॉ", "प्रधानमंत्री", "क्या",
    ],
    "indic-bn": [
        "এবং", "আমি", "তুমি", "এই", "করে", "হয়", "না", "যে", "থেকে", "জন্য", "বাংলা", "ভাষা", "সরকার",
        "কিন্তু", "যদি", "তার", "আমার", "একটি", "দিয়ে", "হবে", "ছিল", "বাংলাদেশ", "ঢাকা", "শ্রী", "ডঃ",
    ],
    "indic-ur": [
        "اور", "کے", "میں", "ہے", "کی", "یہ", "سے", "کو", "نے", "پر", "ہیں", "تھا", "گا", "کہ",
        "جو", "پاکستان", "اردو", "زبان", "حکومت", "لیکن", "اگر", "تو", "ہوں", "گی", "گے", "تھی",
        "تھے", "کر", "یا", "جس", "جن", "جنہیں", "اگرچہ", "مگر", "تاہم", "کیجیے", "کیجئے", "گئیں",
        "تھیں", "خریدا", "ہونگے", "چاہیے", "ہوئیں", "ڈاکٹر",
    ],
}
# Devanagari abbreviations: those indic-nlp-library's splitter knows, and a few it does not.
INDIC_ABBREV = [
    "डॉ", "श्री", "कु", "चि", "सौ", "ए", "बी", "सी", "डी", "ई", "एफ", "जी", "एच", "आई", "जे", "के",
    "एल", "एम", "एन", "ओ", "पी", "क्यू", "आर", "एस", "टी", "यू", "वी", "डब्ल्यू", "एक्स", "वाय",
    "जेड", "क", "ख", "प", "म", "प्र", "मो", "पं", "स्व", "प्रो", "सं", "ई.पू", "रु",
]
# indic_tokenize.triv_tokenizer_indic_pat / triv_tokenizer_urdu_pat, beyond string.punctuation
INDIC_PUNCT = "\u0964\u0965\uaaf1\uaaf0\uabeb\uabec\uabed\uabee\uabef\u1c7e\u1c7f"
URDU_PUNCT = "\u0609\u060a\u060c\u061e\u066a\u066b\u066c\u066d\u06d4"
ELISION_MARKS = ["'", "’", "`", "´", "ʼ", "‘", "′"]
HYPHEN_VARIANTS = ["-", "‐", "‑", "–", "—", "-", "‒", "−"]
ELIDED = [
    "l", "d", "j", "m", "n", "s", "t", "c", "qu", "jusqu", "lorsqu", "puisqu", "quelqu",
    "presqu", "aujourd", "dell", "all", "nell", "sull", "un", "po", "z", "o", "'n", "k", "ch",
]
APOS_TAILS = [
    "s", "t", "n", "ll", "da", "de", "te", "ta", "nin", "ün", "a", "e", "i", "o", "u", "ye", "ya",
    "le", "re", "hui", "un", "il", "en", "est", "lar", "ler", "dan", "den", "in", "ın",
]
NUM_SEPS = [",", ".", " ", "\xa0", "\u202f", "'", "’", "\u066c", "\u060c", "\u02d9", "\u2009"]
NUM_SUFFIXES = [
    "-ban", "-ben", "-án", "-én", "-as", "-es", "-os", "%-kal", "%-os", "-kor", "'da", "'de",
    "'ta", "'te", "'nin", "’da", "'lü", "'ü", ".", ".-án", "%", "‰", "er", "e", "ème", "º", "ª",
    ".º", ".ª", "-й", "-го", "-е", "-х", "-ти", "-ю", "x", "°", "°C", "km", "ga", "an", "en",
    "ste", "ten", "-luvulla", ":n", "-ին", "-ት", "ის",
]
ROMAN = ["I", "II", "III", "IV", "V", "IX", "X", "XI", "XIV", "XIX", "XX", "XXI", "L", "C", "M",
         "MCMXC", "ii", "xiv", "Iv"]
LANG_OPENERS = [*OPENERS, "„", "»", "‹", "›", "«\xa0", "¿", "¡", "‚", "״", "《"]
LANG_CLOSERS = [*CLOSERS, "“", "«", "›", "‹", "\xa0»", "\xa0:", "\xa0!", "\xa0?", "\xa0;", "‘", "״"]
COMMAS = [",", ",", ",", ";", ":", "،", "、", "፣", "؛", "՝", "·"]
WS_TERMINAL = [".", ".", "!", "?", "\n", "…", "։", "।", "。", "؟", "።", "?!", "!!", "...", "\n\n"]
DANDAS = ["।", "।", "॥", "|", ".", "?", "!", "।।", " ।"]
URDU_ENDS = ["۔", "۔", "؟", "!", ".", "،", "؛", " ۔"]
LATIN_PERIODS = [
    "Dr.", "Mr.", "Mrs.", "A.", "B.", "U.S.", "No.", "etc.", "a.m.", "p.m.", "St.", "Ltd.", "3.",
    "10.", "1.5", "v.", "Vol.", "pp.", "e.g.", "i.e.", "A.B.C.", "x.", "Rs.", "Fig.",
]
# fmt: on
SCRIPT_ALIASES = {"Jpan": ["Hira", "Kana", "Hani"], "Kore": ["Hang", "Hani"], "Hans": ["Hani"]}
WIDE_SCRIPTS = {"Hani", "Hira", "Kana", "Hang", "Jpan", "Kore", "Hans", "Hant", "Yiii"}
INHERITED_MARKS = [chr(cp) for cp in range(0x300, 0x370)]
REPEATS = (sre_c.MAX_REPEAT, sre_c.MIN_REPEAT, sre_c.POSSESSIVE_REPEAT)


@functools.cache
def assignments() -> dict[str, tuple[str, str]]:
    """datatrove's (tokenizer class, its language argument) for every code it knows."""
    from datatrove.utils.word_tokenizers import load_tokenizer_assignments

    return {code: (f.args[0], f.args[1] or "") for code, f in load_tokenizer_assignments().items()}


def assignment(lang: str) -> tuple[str, str]:
    a = assignments().get(lang)
    if a is None:
        raise SystemExit(f"datatrove has no word tokenizer for {lang!r}")
    return a


def golden_key(lang: str) -> str:
    """The KEY of golden-KEY.jsonl for `lang`: its spaCy code, "ws" or "indic-CODE"."""
    cls, tok = assignment(lang)
    if cls == "SpaCyTokenizer":
        return tok
    if cls == "WhitespaceTokenizer":
        return "ws"
    if cls == "IndicNLPTokenizer":
        return f"indic-{tok}"
    return f"{cls}-{tok}" if tok else cls


@functools.cache
def assigned_scripts(cls: str, tok: str) -> list[str]:
    """The ISO 15924 scripts of every language datatrove sends to (cls, tok)."""
    from datatrove.utils._import_utils import ASSETS_PATH

    with (Path(ASSETS_PATH) / "tokenizer_assignment.csv").open(encoding="utf-8") as f:
        return sorted(
            {r["script"] for r in csv.DictReader(f) if (r["type"], r["tok_code"]) == (cls, tok)}
        )


@functools.cache
def universe_str() -> str:
    return "".join(_universe())


@functools.cache
def script_pool(iso: str) -> dict[str, list[str]]:
    """Letters ("L", "Lu"), marks, decimal digits and punctuation of script `iso` (scx)."""
    import regex

    out: dict[str, set[str]] = {k: set() for k in ("L", "Lu", "M", "Nd", "P")}
    for name in SCRIPT_ALIASES.get(iso, [iso]):
        try:
            pat = regex.compile(rf"\p{{scx={name}}}")
        except regex.error:
            continue
        for c in pat.findall(universe_str()):
            cat = unicodedata.category(c)
            if cat[0] == "L":
                out["L"].add(c)
                if cat == "Lu":
                    out["Lu"].add(c)
            elif cat[0] in "MP" or cat == "Nd":
                out[cat if cat == "Nd" else cat[0]].add(c)
    if iso in ("Latn", "Cyrl", "Grek", "Armn", "Geor"):
        out["M"].update(INHERITED_MARKS)
    return {k: sorted(v) for k, v in out.items()}


def _children(op: Any, av: Any) -> list[Any]:
    """The sub-patterns of one parsed regex node."""
    if op is sre_c.BRANCH:
        return list(av[1])
    if op is sre_c.SUBPATTERN:
        return [av[3]]
    if op in REPEATS:
        return [av[2]]
    if op in (sre_c.ASSERT, sre_c.ASSERT_NOT):
        return [av[1]]
    if op is sre_c.ATOMIC_GROUP:
        return [av]
    if op is sre_c.GROUPREF_EXISTS:
        return [p for p in av[1:] if p is not None]
    return []


def literal_chars(subs: Iterable[Any]) -> list[str]:
    """The characters parsed patterns spell out: literals and members of small classes."""
    out: set[str] = set()
    stack = list(subs)
    while stack:
        for op, av in stack.pop():
            if op is sre_c.LITERAL:
                out.add(chr(av))
            elif op is sre_c.IN:
                for iop, iav in av:
                    if iop is sre_c.LITERAL:
                        out.add(chr(iav))
                    elif iop is sre_c.RANGE and iav[1] - iav[0] < 64:
                        out.update(
                            chr(c) for c in range(iav[0], iav[1] + 1) if not 0xD800 <= c < 0xE000
                        )
            else:
                stack.extend(_children(op, av))
    return sorted(out)


def in_category(ch: str, code: Any) -> bool:
    if code is sre_c.CATEGORY_DIGIT:
        return unicodedata.category(ch) == "Nd"
    if code is sre_c.CATEGORY_NOT_DIGIT:
        return unicodedata.category(ch) != "Nd"
    if code is sre_c.CATEGORY_SPACE:
        return ch.isspace()
    if code is sre_c.CATEGORY_NOT_SPACE:
        return not ch.isspace()
    word = ch.isalnum() or ch == "_"
    return word if code is sre_c.CATEGORY_WORD else not word


def in_class(ch: str, items: Sequence[tuple[Any, Any]], icase: bool) -> bool:
    """Roughly whether `ch` is in a parsed character class (enough for fuzzing)."""
    forms = {ch, ch.lower(), ch.upper()} if icase else {ch}
    for f in forms:
        if len(f) != 1:
            continue
        o = ord(f)
        for op, av in items:
            if (
                (op is sre_c.LITERAL and av == o)
                or (op is sre_c.RANGE and av[0] <= o <= av[1])
                or (op is sre_c.CATEGORY and in_category(f, av))
            ):
                return True
    return False


def _strip_punct(t: str) -> str:
    i, j = 0, len(t)
    while i < j and unicodedata.category(t[i])[0] in "PS":
        i += 1
    while j > i and unicodedata.category(t[j - 1])[0] in "PS":
        j -= 1
    return t[i:j]


def slice_words(lang: str) -> tuple[list[str], list[str]]:
    """The commonest words of `lang`'s FineWeb-2 slice, and of its tokens glued to punctuation."""
    if not (FW2_DIR / f"{lang}.parquet").exists():
        return [], []
    words: Counter[str] = Counter()
    glued: Counter[str] = Counter()
    for text, _ in read_docs(f"fw2:{lang}"):
        for t in text.split():
            core = _strip_punct(t)
            if "IS_ALPHA" in t or not any(c.isalpha() for c in core):
                continue
            words[core] += 1
            if core != t:
                glued[t] += 1

    def top(c: Counter[str], n: int) -> list[str]:
        return [w for w, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]

    return top(words, 3000), top(glued, 1500)


@functools.cache
def lang_pools(lang: str) -> dict[str, Any]:
    """`lang`'s strings and characters: its tokenizer's rules and patterns, scripts, words."""
    import spacy
    from datatrove.utils.text import TERMINAL_PUNCTUATION
    from spacy.pipeline.sentencizer import Sentencizer

    cls, tok = assignment(lang)
    key = golden_key(lang)
    code = {"SpaCyTokenizer": tok, "WhitespaceTokenizer": "xx"}.get(cls)
    lp: dict[str, Any] = {"cls": cls, "tok": tok, "key": key, "token_match": None}
    lp.update(rules=[], own=[], spaced=[], stop=[], tm_icase=False)
    entries: dict[str, list[Any]] = {"prefix": [], "suffix": [], "infix": []}
    if code:
        nlp = spacy.blank(code)
        base = set(spacy.blank("xx").tokenizer.rules)
        rules = sorted(k for k in nlp.tokenizer.rules if "IS_ALPHA" not in k)
        lp["rules"] = rules
        lp["own"] = [k for k in rules if k not in base]
        lp["spaced"] = [k for k in rules if k.strip() and any(c.isspace() for c in k)]
        lp["stop"] = sorted(nlp.Defaults.stop_words)
        d = nlp.Defaults
        for kind, src in (("prefix", d.prefixes), ("suffix", d.suffixes), ("infix", d.infixes)):
            pats = [getattr(e, "pattern", e) for e in src or ()]
            entries[kind] = [sre_p.parse(s) for s in pats if s.strip()]
        if nlp.tokenizer.token_match is not None:
            pat = nlp.tokenizer.token_match.__self__
            lp["token_match"] = sre_p.parse(pat.pattern, pat.flags)
            lp["tm_icase"] = bool(pat.flags & re.IGNORECASE)
    lp["entries"] = entries
    for kind, subs in entries.items():
        lp[f"{kind}_chars"] = [c for c in literal_chars(subs) if not c.isalnum()]
    if cls == "IndicNLPTokenizer":
        lp["affix_chars"] = sorted(set(string.punctuation + INDIC_PUNCT + URDU_PUNCT + "؟"))
    else:
        lp["affix_chars"] = sorted({*lp["prefix_chars"], *lp["suffix_chars"], *lp["infix_chars"]})

    scripts = assigned_scripts(cls, tok) or ["Latn"]
    primary = lang.split("_", 1)[1] if "_" in lang else scripts[0]
    order = [primary, *(s for s in scripts if s != primary)]
    lp["script_pools"] = {s: script_pool(s) for s in order}
    lp["scripts"] = [s for s in order if lp["script_pools"][s]["L"]] or ["Latn"]
    lp["script_pools"].setdefault("Latn", script_pool("Latn"))
    zeros = sorted(
        {c for s in lp["scripts"] for c in lp["script_pools"][s]["Nd"] if unicodedata.digit(c) == 0}
    )
    lp["digit_sets"] = ["".join(chr(ord(z) + i) for i in range(10)) for z in zeros]

    words, glued = slice_words(lang)
    lp["words"] = [*LANG_WORDS.get(key, []), *words]
    lp["raw"] = glued
    lp["extras"] = LANG_EXTRAS.get(key, [])
    seen = [*lp["stop"], *lp["own"], *lp["words"], *lp["extras"]]
    letters = set(lp["script_pools"][lp["scripts"][0]]["L"])  # the primary script's
    lp["alphabet"] = sorted({c for w in seen for c in w if c in letters})
    lp["marks"] = sorted({c for w in seen for c in w if unicodedata.category(c)[0] == "M"})
    sent = set(Sentencizer.default_punct_chars)
    script_ends = {c for s in lp["scripts"] for c in lp["script_pools"][s]["P"] if c in sent}
    if cls == "IndicNLPTokenizer":
        lp["enders"] = URDU_ENDS if tok == "ur" else DANDAS
    else:
        lp["enders"] = [".", ".", ".", "!", "?", "...", "…", ":", "", "?!", *sorted(script_ends)]
    lp["terminal"] = sorted(TERMINAL_PUNCTUATION)
    lp["upper"] = sorted(
        {c for s in lp["scripts"] for c in lp["script_pools"][s]["Lu"]}
        | set(string.ascii_uppercase)
    )
    if key == "indic-hi":
        lp["abbrev"] = INDIC_ABBREV
    elif key == "indic-bn":  # the same letters, by Brahmi offset, as the splitter transliterates
        bn = [
            "".join(chr(ord(c) + 0x80) if "\u0900" <= c <= "\u097f" else c for c in a)
            for a in INDIC_ABBREV
        ]
        lp["abbrev"] = [a for a in bn if all(unicodedata.category(c) != "Cn" for c in a)]
    else:
        lp["abbrev"] = []
    return lp


class LangFuzzer(Fuzzer):
    """Texts aimed at one language: its tokenizer's rules and patterns, scripts and words."""

    def __init__(self, seed: int, golden: bool, lang: str) -> None:
        self.lang = lang
        self.lp = lang_pools(lang)
        super().__init__(seed, golden)

    def piece_weights(self) -> list[tuple[Callable[[], str], float]]:
        lp = self.lp
        has = {k: bool(v) for k, v in lp["entries"].items()}
        q = 0.25 if self.golden else 1  # Unicode noise: the English golden covers it already
        weights: list[tuple[Callable[[], str], float]] = [
            (self.word, 22),
            (self.rule, 14 if lp["rules"] else 0),
            (self.own_rule, 10 if lp["own"] else 0),
            (self.spaced_rule, 3 if lp["spaced"] else 0),
            (self.prefixed, 5 if has["prefix"] else 0),
            (self.suffixed, 6 if has["suffix"] else 0),
            (self.infixed, 6 if has["infix"] else 0),
            (self.matched, 8 if lp["token_match"] else 0),
            (self.extra, 5 if lp["extras"] else 0),
            (self.raw_token, 6 if lp["raw"] else 0),
            (self.elision, 4),
            (self.hyphenated, 3),
            (self.number, 4),
            (self.script_number, 4),
            (self.url, 2.5),
            (self.email, 1),
            (self.native, 4),
            (self.noise, 3),
            (self.sent_end, 6),
            (self.uni_punct, 2 * q),
            (self.emoji, 1.5 * q),
            (self.control, 1 * q),
            (self.random_cp, 1 * q),
            (self.ellipsis, 1.5),
            (self.lang_affixed, 5),
            (self.punct_run, 2 * q),
            (self.combining, 2 * q),
            (self.cjk, 0.5 * q),
            (self.emoticon, 1),
            (self.new_chars, 0.3 * q),
        ]
        return [(fn, w) for fn, w in weights if w]

    def text_weights(self) -> list[tuple[Callable[[], str], float]]:
        return [
            (self.t_space, 2),
            (self.t_punct, 2),
            (self.t_single, 30),
            (self.t_glued, 16),
            (self.t_mixed, 18),
            (self.t_prose, 16),
            (self.t_sentences, 6),
            (self.t_boundary, 6),
            # Long glued runs cost spaCy quadratic time (~10 s for 5 kB): 90% of a recording's.
            (self.t_long, 0 if self.golden else 0.5),
            (self.t_soup, 0.1 if self.golden else 0.4),
        ]

    def target_len(self) -> int:
        if not self.golden:
            return super().target_len()
        r = self.rng.random()  # shorter than the English golden: ~60 small files
        if r < 0.7:
            return self.rng.randint(1, 10)
        if r < 0.97:
            return self.rng.randint(10, 30)
        return self.rng.randint(30, 70)

    # language helpers
    def script(self) -> str:
        s = self.lp["scripts"]
        return s[0] if len(s) == 1 or self.chance(0.75) else self.pick(s[1:])

    def letter(self) -> str:
        """One letter of the language: its own alphabet mostly, else its scripts'."""
        iso = self.script()
        if iso == self.lp["scripts"][0] and self.lp["alphabet"] and self.chance(0.85):
            return self.pick(self.lp["alphabet"])
        return self.pick(self.lp["script_pools"][iso]["L"])

    def native(self) -> str:
        """Letters of the language, now and then followed by a combining mark."""
        iso = self.script()
        marks = self.lp["script_pools"][iso]["M"] or self.lp["marks"]
        out = []
        for _ in range(self.rng.randint(1, 3 if iso in WIDE_SCRIPTS else 8)):
            out.append(self.letter())
            if marks and self.chance(0.12):
                out.append(self.pick(marks))
        return "".join(out)

    def native_char(self) -> str:
        r = self.rng.random()
        if r < 0.7:
            return self.letter()
        if r < 0.8:
            return self.digits(1)
        if r < 0.9 and self.lp["affix_chars"]:
            return self.pick(self.lp["affix_chars"])
        return self.pick(self.p["P"])

    def force_case(self, s: str) -> str:
        return self.rng.choice([str.upper, str.lower, str.title, str.swapcase, str.capitalize])(s)

    def flip_case(self, ch: str) -> str:
        if self.chance(0.35):
            f = ch.swapcase()
            if len(f) == 1:
                return f
        return ch

    # regex sampling: strings the tokenizer's own patterns match
    def sample(self, sub: Any, icase: bool = False, groups: dict[int, str] | None = None) -> str:
        """A random string the parsed regex `sub` matches; lookarounds are spelled out too."""
        groups = {} if groups is None else groups
        return "".join(self.sample_op(op, av, icase, groups) for op, av in sub)

    def sample_op(self, op: Any, av: Any, icase: bool, groups: dict[int, str]) -> str:
        if op is sre_c.LITERAL:
            return self.flip_case(chr(av)) if icase else chr(av)
        if op is sre_c.NOT_LITERAL:
            return self.unlike([(sre_c.LITERAL, av)], icase)
        if op is sre_c.ANY:
            return self.native_char()
        if op is sre_c.IN:
            return self.sample_in(av, icase)
        if op is sre_c.CATEGORY:
            return self.sample_category(av)
        if op is sre_c.BRANCH:
            return self.sample(self.rng.choice(av[1]), icase, groups)
        if op is sre_c.SUBPATTERN:
            group, add_flags, _, sub = av
            s = self.sample(sub, icase or bool(add_flags & re.IGNORECASE), groups)
            if group is not None:
                groups[group] = s
            return s
        if op in REPEATS:
            lo, hi, sub = av
            n = min(lo + self.rng.choice((0, 0, 1, 1, 2, 3)), hi)
            return "".join(self.sample(sub, icase, groups) for _ in range(n))
        if op is sre_c.ATOMIC_GROUP:
            return self.sample(av, icase, groups)
        if op is sre_c.ASSERT:
            return self.sample(av[1], icase, groups) if self.chance(0.85) else ""
        if op is sre_c.ASSERT_NOT:
            return self.sample(av[1], icase, groups) if self.chance(0.1) else ""
        if op is sre_c.GROUPREF:
            return groups.get(av, "")
        if op is sre_c.GROUPREF_EXISTS:
            group, yes, no = av
            sub = yes if group in groups else no
            return self.sample(sub, icase, groups) if sub is not None else ""
        return ""  # anchors

    def sample_in(self, items: Sequence[tuple[Any, Any]], icase: bool) -> str:
        if items and items[0][0] is sre_c.NEGATE:
            return self.unlike(items[1:], icase)
        if self.chance(0.5):  # prefer the language's own characters
            for _ in range(6):
                ch = self.native_char()
                if in_class(ch, items, icase):
                    return self.flip_case(ch) if icase else ch
        op, av = self.rng.choice(items)
        if op is sre_c.LITERAL:
            ch = chr(av)
        elif op is sre_c.RANGE:
            cp = self.rng.randint(av[0], av[1])
            ch = chr(av[0] if 0xD800 <= cp < 0xE000 else cp)
        elif op is sre_c.CATEGORY:
            ch = self.sample_category(av)
        else:
            ch = "?"
        return self.flip_case(ch) if icase else ch

    def unlike(self, items: Sequence[tuple[Any, Any]], icase: bool) -> str:
        for _ in range(12):
            ch = self.native_char() if self.chance(0.6) else self.pick(self.p["P"] + self.p["S"])
            if not in_class(ch, items, icase):
                return ch
        return self.pick(self.p["S"])

    def sample_category(self, code: Any) -> str:
        if code is sre_c.CATEGORY_DIGIT:
            return self.digits(1)
        if code is sre_c.CATEGORY_SPACE:
            return self.pick([" ", " ", " ", "\t", "\n", "\xa0"])
        if code is sre_c.CATEGORY_WORD:
            return self.letter() if self.chance(0.8) else self.digits(1)
        if code is sre_c.CATEGORY_NOT_WORD:
            return self.pick(self.p["P"])
        return self.letter()

    def affix(self, kind: str) -> str:
        """A string one of the tokenizer's prefix / suffix / infix patterns matches."""
        entries = self.lp["entries"][kind]
        return self.sample(self.rng.choice(entries)) if entries else ""

    # pieces
    def word(self) -> str:
        lp = self.lp
        r = self.rng.random()
        if r < 0.35 and lp["words"]:
            s = self.pick(lp["words"])
        elif r < 0.6 and lp["stop"]:
            s = self.pick(lp["stop"])
        elif r < 0.86:
            s = self.native()
        elif r < 0.94:
            s = self.pick(ENGLISH)
        else:
            s = self.letters()
        return self.recase(s)

    def digits(self, n: int | None = None) -> str:
        sets = self.lp["digit_sets"]
        if sets and self.chance(0.2):
            ds = self.pick(sets)
            return "".join(self.pick(ds) for _ in range(n or self.rng.randint(1, 5)))
        return super().digits(n)

    def noise(self) -> str:
        """English-ish words, as found on every language's web pages."""
        return super().word() if self.chance(0.7) else self.extra_word()

    def rule(self) -> str:
        return self.vary(self.pick(self.lp["rules"]))

    def own_rule(self) -> str:
        """One of the language's own exceptions (not those every spaCy language has)."""
        return self.vary(self.pick(self.lp["own"]))

    def spaced_rule(self) -> str:
        """An exception spanning white space (spaCy's special-case matcher handles those)."""
        return self.vary(self.pick(self.lp["spaced"]))

    def vary(self, k: str) -> str:
        """A rule key, maybe re-cased, re-punctuated or mangled, in context."""
        r = self.rng.random()
        if r < 0.14:
            k = self.force_case(k)
        elif r < 0.24:
            for a in ELISION_MARKS:
                if a in k:
                    k = k.replace(a, self.pick(ELISION_MARKS))
        elif r < 0.3:
            for h in HYPHEN_VARIANTS:
                if h in k:
                    k = k.replace(h, self.pick(HYPHEN_VARIANTS))
        elif r < 0.34 and len(k) > 1:
            i = self.rng.randrange(len(k))
            k = k[:i] + k[i + 1 :] if self.chance(0.5) else k[:i] + k[i] + k[i:]
        elif r < 0.38 and k.endswith("."):
            k = k[:-1]
        elif r < 0.41:
            k += "."
        elif r < 0.45 and " " in k:
            k = k.replace(" ", self.pick(["", "\xa0", "  ", "\n", "\t", "\u202f"]))
        return self.lang_context(k)

    def lang_context(self, k: str) -> str:
        """`k` with brackets, the language's prefixes / suffixes, or glued words around it."""
        r = self.rng.random()
        if r < 0.12:
            i = self.rng.randrange(len(OPENERS))
            k = OPENERS[i] + k + (CLOSERS[i] if self.chance(0.8) else self.pick(LANG_CLOSERS))
        elif r < 0.2:
            k = self.affix("prefix") + k
        elif r < 0.25:
            k = self.pick(LANG_OPENERS + self.p["prefix_only"] + self.p["currency"]) + k
        if self.chance(0.22):
            k += self.affix("suffix")
        elif self.chance(0.12):
            k += self.pick(TRAILERS)
        if self.chance(0.06):
            k = (self.word() if self.chance(0.6) else self.digits()) + k
        if self.chance(0.06):
            k += self.word() if self.chance(0.6) else self.digits()
        if self.chance(0.03) and self.lp["rules"]:
            k += self.pick(self.lp["rules"])
        return k

    def core(self) -> str:
        r = self.rng.random()
        if r < 0.5:
            return self.word()
        if r < 0.7:
            return self.native()[: self.rng.randint(1, 3)]
        if r < 0.85 or not self.lp["rules"]:
            return self.digits()
        return self.pick(self.lp["rules"])

    def prefixed(self) -> str:
        pre = self.affix("prefix") + (self.affix("prefix") if self.chance(0.2) else "")
        return pre + self.core()

    def suffixed(self) -> str:
        suf = self.affix("suffix") + (self.affix("suffix") if self.chance(0.25) else "")
        return self.core() + suf

    def infixed(self) -> str:
        s = self.core() + self.affix("infix") + self.core()
        return s + self.affix("infix") + self.core() if self.chance(0.2) else s

    def matched(self) -> str:
        """A string the language's token_match accepts (fr, hu, tr)."""
        s = self.sample(self.lp["token_match"], self.lp["tm_icase"])
        return self.lang_context(s) if self.chance(0.25) else s

    def extra(self) -> str:
        s = self.pick(self.lp["extras"])
        return self.lang_context(s) if self.chance(0.6) else s

    def raw_token(self) -> str:
        """A token glued to punctuation, as found in the language's FineWeb-2 slice."""
        return self.pick(self.lp["raw"])

    def elision(self) -> str:
        a = self.pick(ELISION_MARKS)
        r = self.rng.random()
        if r < 0.4:
            s = self.recase(self.pick(ELIDED)) + a + self.word()
        elif r < 0.6:
            s = self.word() + a + self.pick(APOS_TAILS)
        elif r < 0.75:
            s = a + self.word()
        elif r < 0.85:
            s = self.word() + a
        else:
            s = self.word() + a + self.word() + a + self.word()
        return self.lang_context(s) if self.chance(0.25) else s

    def script_number(self) -> str:
        """Numbers in the language's digits: separators, ordinals, dates, inflections."""
        rng = self.rng
        sets = self.lp["digit_sets"]
        ds = self.pick(sets) if sets and self.chance(0.6) else "0123456789"

        def d(n: int) -> str:
            return "".join(self.pick(ds) for _ in range(n))

        r = rng.random()
        if r < 0.22:
            groups = "".join(self.pick(NUM_SEPS) + d(3) for _ in range(rng.randint(1, 2)))
            s = d(rng.randint(1, 3)) + groups
        elif r < 0.36:
            s = d(rng.randint(1, 4)) + self.pick([".", ",", "٫", "·", "'"]) + d(rng.randint(1, 3))
        elif r < 0.52:
            s = d(rng.randint(1, 4)) + self.pick(NUM_SUFFIXES)
        elif r < 0.6:
            s = self.pick(["№", "№ ", "No.", "Nr.", "nr.", "n°", "N°", "#", "§", "§ ", "Art."])
            s += d(rng.randint(1, 3))
        elif r < 0.7:
            sep = self.pick([".", "/", "-", ". "])
            s = d(rng.randint(1, 2)) + sep + d(rng.randint(1, 2)) + sep + d(4)
        elif r < 0.78:
            s = d(rng.randint(1, 2)) + ":" + d(2)
            s += self.pick(NUM_SUFFIXES) if self.chance(0.3) else ""
        elif r < 0.86:
            s = self.pick(ROMAN) + self.pick([".", "", "-", "e", ".-", "-ban", "'da", "-й"])
        elif r < 0.93:
            c = self.pick(self.p["currency"])
            n = d(rng.randint(1, 4))
            s = c + n if self.chance(0.5) else n + self.pick(["", " ", "\xa0"]) + c
        else:
            s = d(rng.randint(1, 3)) + self.pick(
                ["%", " %", "\xa0%", "‰", "°", "°C", "km", "m²", "kg", "€", "$", "x"]
            )
        return self.lang_context(s) if self.chance(0.2) else s

    def lang_affixed(self) -> str:
        """Characters the language's affix patterns name, glued around a word or number."""
        pre_pool = self.lp["prefix_chars"] or self.p["punct"]
        suf_pool = self.lp["suffix_chars"] or self.p["punct"]
        pre = "".join(self.pick(pre_pool) for _ in range(self.rng.choice([0, 1, 1, 2])))
        suf = "".join(self.pick(suf_pool) for _ in range(self.rng.choice([0, 1, 1, 2, 3])))
        return pre + (self.word() if self.chance(0.7) else self.script_number()) + suf

    # whole texts
    def prose_word(self) -> str:
        lp = self.lp
        r = self.rng.random()
        if r < 0.55:
            return self.word()
        if r < 0.7 and lp["rules"]:
            return self.pick(lp["own"] or lp["rules"])
        if r < 0.78 and lp["extras"]:
            return self.pick(lp["extras"])
        if r < 0.86 and lp["raw"]:
            return self.pick(lp["raw"])
        return self.piece()

    def t_prose(self) -> str:
        """Prose-like text: capitalised sentences, the language's punctuation, numbers, links."""
        n = self.target_len()
        out: list[str] = []
        total = 0
        while total < n:
            words = [
                self.prose_word() for _ in range(self.rng.randint(1, 5 if self.golden else 12))
            ]
            words[0] = words[0][:1].upper() + words[0][1:]
            for i in range(1, len(words)):
                r = self.rng.random()
                if r < 0.08:
                    words[i - 1] += self.pick(COMMAS)
                elif r < 0.12:
                    words[i] = self.pick(LANG_OPENERS) + words[i]
                elif r < 0.16:
                    words[i] += self.pick(LANG_CLOSERS)
                elif r < 0.19:
                    words[i] = self.script_number()
                elif r < 0.21:
                    words[i] = self.url() if self.chance(0.7) else self.email()
                elif r < 0.24:
                    words[i] = self.hyphenated() if self.chance(0.5) else self.elision()
            s = " ".join(words) + self.pick(self.lp["enders"])
            out.append(s)
            total += len(s) + 1
        joins = [" ", " ", " ", " ", "\n", "  ", "\n\n", "", "\xa0"]
        return "".join(s + self.pick(joins) for s in out)


class WsFuzzer(LangFuzzer):
    """Texts for datatrove's WhitespaceTokenizer: spaCy "xx" words, a sentence regex."""

    def piece_weights(self) -> list[tuple[Callable[[], str], float]]:
        return [*super().piece_weights(), (self.terminal, 8), (self.acronym, 6)]

    def text_weights(self) -> list[tuple[Callable[[], str], float]]:
        return [*super().text_weights(), (self.t_ws, 24)]

    def end(self) -> str:
        """Terminal punctuation, maybe repeated, maybe followed by a quote."""
        t = self.pick(WS_TERMINAL) if self.chance(0.5) else self.pick(self.lp["terminal"])
        if self.chance(0.15):
            t *= self.rng.randint(2, 3)
        if self.chance(0.2):
            t += self.pick(['"', "'", "”", "’", "»", ")", "\"'", "”."])
        return t

    def terminal(self) -> str:
        return self.word() + self.end()

    def acronym(self) -> str:
        """Upper-case letter + period runs, which the sentence regex must not split."""
        s = "".join(self.pick(self.lp["upper"]) + "." for _ in range(self.rng.randint(1, 4)))
        r = self.rng.random()
        if r < 0.2:
            s = s[:-1]
        elif r < 0.3:
            s = s.lower()
        return s + self.end() if self.chance(0.3) else s

    def t_ws(self) -> str:
        out = []
        for _ in range(self.rng.randint(1, 3 if self.golden else 8)):
            words = [
                self.word() if self.chance(0.75) else self.piece()
                for _ in range(self.rng.randint(0, 4 if self.golden else 10))
            ]
            out.append(" ".join(words) + (self.end() if self.chance(0.75) else ""))
        joins = ["", " ", " ", "\n", "\n\n", "\r\n", "  ", "\t", "\u2028", "\xa0"]
        return "".join(o + self.pick(joins) for o in out)


class IndicFuzzer(LangFuzzer):
    """Texts for datatrove's IndicNLPTokenizer: indic-nlp-library's tokenizer and splitter."""

    def piece_weights(self) -> list[tuple[Callable[[], str], float]]:
        lp = self.lp
        weights: list[tuple[Callable[[], str], float]] = [
            (self.word, 30),
            (self.indic_punct, 10),
            (self.danda, 8),
            (self.indic_number, 10),
            (self.abbrev, 8 if lp["abbrev"] else 0),
            (self.latin_period, 6),
            (self.raw_token, 6 if lp["raw"] else 0),
            (self.script_number, 4),
            (self.url, 2),
            (self.email, 1),
            (self.native, 6),
            (self.noise, 4),
            (self.sent_end, 5),
            (self.uni_punct, 2),
            (self.combining, 2),
            (self.emoji, 1),
            (self.control, 1),
            (self.random_cp, 1),
            (self.ellipsis, 1.5),
            (self.punct_run, 3),
            (self.elision, 1),
            (self.hyphenated, 2),
        ]
        return [(fn, w) for fn, w in weights if w]

    def text_weights(self) -> list[tuple[Callable[[], str], float]]:
        return [
            (self.t_space, 2),
            (self.t_punct, 2),
            (self.t_single, 22),
            (self.t_glued, 14),
            (self.t_mixed, 18),
            (self.t_prose, 22),
            (self.t_sentences, 8),
            (self.t_boundary, 8),
            (self.t_long, 0 if self.golden else 1.5),
            (self.t_soup, 0.1 if self.golden else 0.4),
        ]

    def indic_punct(self) -> str:
        """The trivial tokenizer's punctuation (string.punctuation, dandas, Urdu marks)."""
        p = "".join(self.pick(self.lp["affix_chars"]) for _ in range(self.rng.randint(1, 2)))
        r = self.rng.random()
        if r < 0.4:
            return self.word() + p
        if r < 0.6:
            return p + self.word()
        if r < 0.85:
            return self.word() + p + self.word()
        return p

    def danda(self) -> str:
        return self.word() + (" " if self.chance(0.2) else "") + self.pick(self.lp["enders"])

    def indic_number(self) -> str:
        """Numbers the tokenizer re-joins: `([0-9]+ [,.:/] )+[0-9]+` after splitting."""
        rng = self.rng

        def num() -> str:
            return "".join(self.pick("0123456789") for _ in range(rng.randint(1, 4)))

        k = rng.randint(1, 3)
        r = rng.random()
        if r < 0.5:
            return num() + "".join(self.pick(",.:/") + num() for _ in range(k))
        if r < 0.75:
            return num() + "".join(" " + self.pick(",.:/") + " " + num() for _ in range(k))
        if r < 0.9:
            return self.script_number()
        return num() + self.pick(",.:/")

    def abbrev(self) -> str:
        """Abbreviations the sentence splitter keeps sentences together at, and others."""
        rng = self.rng
        a = self.pick(self.lp["abbrev"]) + "."
        r = rng.random()
        if r < 0.4:
            return a + " " + self.word()
        if r < 0.6:
            return a + self.word()
        if r < 0.8:
            return " ".join(self.pick(self.lp["abbrev"]) + "." for _ in range(rng.randint(2, 3)))
        return a

    def latin_period(self) -> str:
        return self.pick(LATIN_PERIODS)


def make_fuzzer(lang: str, seed: int, golden: bool = False) -> Fuzzer:
    """The fuzzer for datatrove language code `lang` ("eng": the English one)."""
    if lang == "eng":
        return Fuzzer(seed, golden)
    cls, _ = assignment(lang)
    if cls == "WhitespaceTokenizer":
        return WsFuzzer(seed, golden, lang)
    if cls == "IndicNLPTokenizer":
        return IndicFuzzer(seed, golden, lang)
    return LangFuzzer(seed, golden, lang)


# --------------------------------------------------------------------------
# Multilingual goldens: edge cases, the FineWeb-2 slice (if any), selected fuzz


def golden_slice_items(lang: str) -> list[Item]:
    """Short whole documents, C4-style lines and random excerpts of `lang`'s FineWeb-2 slice."""
    if not (FW2_DIR / f"{lang}.parquet").exists():
        return []
    docs = [(t, i) for t, i in read_docs(f"fw2:{lang}") if "IS_ALPHA" not in t]
    shortest = sorted(docs, key=lambda d: (len(d[0]), d[1]))[:3]
    items: list[Item] = [(t, {"k": "doc", "id": i}) for t, i in shortest]
    seen = {t for t, _ in items}
    lines: list[tuple[str, frozenset[str]]] = []
    for t, _ in docs:
        for line in c4_lines(t):
            if 0 < len(line) <= 120 and line not in seen:
                seen.add(line)
                lines.append((line, features(line)))
    items.extend((lines[i][0], {"k": "line"}) for i in greedy(lines, 80, set()))
    rng = random.Random(GOLDEN_SEED)
    for _ in range(80):  # cut anywhere: mid-word, mid-cluster, mid-line
        t = docs[rng.randrange(len(docs))][0]
        a = rng.randrange(len(t))
        x = t[a : a + rng.randint(8, 120)]
        if x not in seen:
            seen.add(x)
            items.append((x, {"k": "excerpt"}))
    return items


def char_class(c: str, keep: frozenset[str]) -> str:
    """`c` itself if punctuation the tokenizer's patterns name (`keep`), else its class.

    Letters are told apart by script only for the scripts named in `keep` (as
    "script:NAME"), so that rows of rare scripts do not crowd out the language.
    """
    if c in keep or (c.isascii() and not c.isalnum()):
        return c
    cat = unicodedata.category(c)
    if cat[0] in "ZC":
        return c
    if cat[0] == "L":
        name = unicodedata.name(c, "?").split(" ")[0]
        return f"{cat}:{name if 'script:' + name in keep else 'other'}"
    return cat


def shape(w: str, keep: frozenset[str]) -> str:
    """A token's shape: x X d m o n p s per character class, named punctuation as is."""
    out: list[str] = []
    for c in w[:16]:
        cat = unicodedata.category(c)
        if c in keep or (c.isascii() and not c.isalnum()):
            k = c
        else:
            k = {"Ll": "x", "Lu": "X", "Nd": "d"}.get(cat) or {
                "M": "m",
                "L": "o",
                "N": "n",
                "P": "p",
                "S": "s",
            }.get(cat[0], c)
        if out[-2:] != [k, k]:
            out.append(k)
    return "".join(out)


def row_features(row: dict[str, Any], keep: frozenset[str], rules: set[str]) -> frozenset[str]:
    """What a recorded row exercises: split points between glued tokens, token shapes."""
    f = {f"raise:{row['raise']}:{row['at']}"} if "raise" in row else set()
    if "s" in row:
        f.add(f"s:{min(row['s'], 4)}")
    text = row["t"]
    end: int | None = None
    for w in row.get("w", ()):
        i = text.find(w, end or 0)
        if i < 0:
            f.add("lost")
            continue
        glued = end is not None and i == end
        if glued:
            f.add(f"glue:{char_class(text[i - 1], keep)}|{char_class(w[0], keep)}")
        f.add(("rule:" if w in rules else "tok:") + shape(w, keep))
        end = i + len(w)
    f.update("c:" + char_class(c, keep) for c in set(text))
    return frozenset(f)


def select_rows(lines: list[str], keep: frozenset[str], rules: set[str]) -> list[int]:
    """Short rows within the size budget: a greedy cover of what rows exercise, then random."""
    sizes = [len(line.encode()) + 1 for line in lines]
    cands = [
        (line, row_features(json.loads(line), keep, rules) if n <= 300 else frozenset())
        for line, n in zip(lines, sizes, strict=True)
    ]
    k, budget = GOLDEN_LANG_FUZZ, GOLDEN_LANG_BYTES
    chosen: list[int] = []
    used = 0
    for i in greedy(cands, k // 2, set(), scale=20):
        if used + sizes[i] > budget // 2:
            break
        chosen.append(i)
        used += sizes[i]
    taken = set(chosen)
    short = max(90, sorted(sizes)[len(sizes) // 3])  # rows that raise are longer
    rest = [i for i in range(len(lines)) if i not in taken and sizes[i] <= short]
    random.Random(GOLDEN_SEED).shuffle(rest)
    for i in rest:
        if len(chosen) == k or used + sizes[i] > budget:
            break
        chosen.append(i)
        used += sizes[i]
    return sorted(chosen)


def golden_lang(out: Path, workers: int) -> None:
    """The committed golden for `LANG`; see golden_slice_items and select_rows."""
    t0 = time.monotonic()
    fixed: list[Item] = []  # (the English golden and every fuzz recording have EDGE)
    fixed += golden_slice_items(LANG)
    f = make_fuzzer(LANG, GOLDEN_SEED, golden=True)
    seen = {t for t, _ in fixed}
    cands: list[Item] = []
    while len(cands) < GOLDEN_LANG_FUZZ * GOLDEN_LANG_CANDIDATES:
        s, _ = f.text()
        if any(0xD800 <= ord(c) <= 0xDFFF for c in s):
            raise AssertionError("lone surrogate generated")
        if s not in seen:
            seen.add(s)
            cands.append((s, {}))
    lines = [line for line, _ in recorded([*fixed, *cands], workers, 16)]
    head, tail = lines[: len(fixed)], lines[len(fixed) :]
    lp = lang_pools(LANG)
    scripts = {"script:" + unicodedata.name(c, "?").split(" ")[0] for c in lp["alphabet"]}
    named = frozenset([*lp["affix_chars"], *scripts])
    rows = head + [tail[i] for i in select_rows(tail, named, set(lp["rules"]))]
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(line + "\n" for line in rows)
    raises = Counter(r["raise"] for r in map(json.loads, rows) if "raise" in r)
    report(out, len(rows), raises, t0)


def goldens(out_dir: Path, keys: list[str], workers: int) -> None:
    """golden-KEY.jsonl for each key (default: the spaCy ones and "ws"), one process each."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in keys or [k for k in REPRESENTATIVE if not k.startswith("indic-")]:
        out = out_dir / f"golden-{key}.jsonl"
        lang = REPRESENTATIVE[key]
        cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "--workers", str(workers)]
        subprocess.run([*cmd, "--lang", lang, "golden", str(out)], check=True)


# --------------------------------------------------------------------------
# FineWeb-2 volume per datatrove tokenizer

FW2_DISTRIBUTION = (
    "https://raw.githubusercontent.com/huggingface/fineweb-2/main/"
    "fineweb2-language-distribution.csv"
)
NOT_PORTABLE = {"zh", "ja", "vi"}  # spaCy languages backed by jieba / sudachi / pyvi


def language_table(dist: Path, out: Path) -> None:
    """FineWeb-2's train subsets by datatrove tokenizer: subsets, words, UTF-8 bytes."""
    from importlib.metadata import version

    table = assignments()
    with dist.open(encoding="utf-8") as fh:
        rows = [
            r
            for r in csv.DictReader(fh)
            if r["split"] == "train" and not r["subset"].endswith("_removed")
        ]

    def num(x: str) -> int:
        return int(x) if x.strip().isdigit() else 0

    subsets = []
    for r in rows:
        cls, tok = table.get(r["subset"], ("", ""))
        subsets.append(
            {
                "subset": r["subset"],
                "name": r["name"],
                "class": cls or None,
                "tok": tok or None,
                "words": num(r["words"]),
                "documents": num(r["documents"]),
                "utf8_bytes": num(r["utf8_bytes"]),
            }
        )
    total_bytes = sum(s["utf8_bytes"] for s in subsets)
    total_words = sum(s["words"] for s in subsets)

    def tokenizer(s: dict[str, Any]) -> str:
        if not s["class"]:
            return "(unassigned)"
        return f"{s['class']}:{s['tok']}" if s["tok"] else s["class"]

    def portable(s: dict[str, Any]) -> str:
        if s["class"] == "SpaCyTokenizer":
            return "spacy-not-portable" if s["tok"] in NOT_PORTABLE else "spacy-rule-based"
        return s["class"] or "(unassigned)"

    def agg(key: Callable[[dict[str, Any]], str]) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for s in subsets:
            g = groups.setdefault(
                key(s), {"subsets": 0, "words": 0, "documents": 0, "utf8_bytes": 0}
            )
            g["subsets"] += 1
            for k in ("words", "documents", "utf8_bytes"):
                g[k] += s[k]
        res = [
            {"key": k, **g, "byte_share": g["utf8_bytes"] / total_bytes} for k, g in groups.items()
        ]
        return sorted(res, key=lambda g: (-g["utf8_bytes"], g["key"]))

    doc = {
        "source": FW2_DISTRIBUTION,
        "split": "train (subsets without _removed)",
        "datatrove": version("datatrove"),
        "subsets": len(subsets),
        "words": total_words,
        "utf8_bytes": total_bytes,
        "by_class": agg(lambda s: s["class"] or "(unassigned)"),
        "by_backend": agg(portable),
        "by_tokenizer": agg(tokenizer),
        "unassigned": sorted(
            (s for s in subsets if not s["class"]), key=lambda s: (-s["utf8_bytes"], s["subset"])
        ),
        "per_subset": sorted(subsets, key=lambda s: (-s["utf8_bytes"], s["subset"])),
    }
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    def md(rows: list[dict[str, Any]], title: str) -> str:
        head = f"| {title} | subsets | words (G) | UTF-8 (GB) | share |\n|---|--:|--:|--:|--:|\n"
        return head + "".join(
            f"| {g['key']} | {g['subsets']} | {g['words'] / 1e9:.1f} | "
            f"{g['utf8_bytes'] / 1e9:.1f} | {100 * g['byte_share']:.2f}% |\n"
            for g in rows
        )

    sys.stdout.write(
        f"FineWeb-2 train: {len(subsets)} subsets, {total_words / 1e9:.1f} G words, "
        f"{total_bytes / 1e9:.1f} GB UTF-8; datatrove {doc['datatrove']}\n\n"
        + md(doc["by_backend"], "backend")
        + "\n"
        + md(doc["by_class"], "class")
        + "\n"
        + md(doc["by_tokenizer"][:25], "tokenizer (top 25)")
        + f"\nunassigned: {len(doc['unassigned'])} subsets, "
        f"{sum(s['utf8_bytes'] for s in doc['unassigned']) / 1e9:.2f} GB\n"
    )
    log(f"wrote {out}")


# --------------------------------------------------------------------------
# IS_ALPHA

# spaCy's StringStore maps "IS_ALPHA" to symbol id 1, which preshed reserves as
# its DELETED key. Once a memory zone that saw a token "IS_ALPHA" has closed,
# every later attempt to make that lexeme fails, so datatrove's retry (drop
# every "IS_ALPHA" substring and tokenize again) kicks in.
ISALPHA_CASES = [
    "IS_ALPHA",
    "IS_ALPHA.",
    "foo IS_ALPHA bar",
    "(IS_ALPHA)",
    '"IS_ALPHA"',
    "IS_ALPHA's",
    "foo-IS_ALPHA",
    "xIS_ALPHAx",
    "IS_ALPHAs",
    "is_alpha",
    "IS_ALPHA IS_ALPHA",
    "IS_ALPHA\nIS_ALPHA",
    "Hello. IS_ALPHA! World? Done.",
    "IS_ALPHA IS_IS_ALPHAALPHA",
    "IS_IS_ALPHAALPHA",
    "IS_ASCII LIKE_URL ORTH",
]
UNRELATED = ["Hello world.", "The quick brown fox. It jumps!", "IS_ALPHAx", "x", ""]


def probe(mode: str) -> None:
    """Child process: run `mode`'s history, then record the case from stdin."""
    from datatrove.utils.text import split_into_parts

    text = json.loads(sys.stdin.read())
    if mode in ("after_unrelated", "after_isalpha"):
        for t in UNRELATED:
            record(t)
    if mode == "after_isalpha":
        record("prime IS_ALPHA")
    split_into_parts.cache_clear()  # datatrove's lru_cache(5) would hide the effect
    first = record(text)
    split_into_parts.cache_clear()
    again = record(text)
    sys.stdout.write(json.dumps({"first": first, "again": again}, ensure_ascii=False))


def isalpha(out: Path | None) -> None:
    modes = ["fresh", "after_unrelated", "after_isalpha"]
    jobs = [(t, m) for t in ISALPHA_CASES for m in modes]

    def go(job: tuple[str, str]) -> dict[str, Any]:
        text, mode = job
        r = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).resolve()), "--lang", LANG, "_probe", mode],
            input=json.dumps(text),
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(r.stdout)

    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(go, jobs))

    def brief(row: dict[str, Any]) -> str:
        if "raise" in row:
            return f"raise {row['raise']} (in {row['at']})"
        return f"w={row['w']} s={row['s']}"

    rows = []
    for (text, mode), res in zip(jobs, results, strict=True):
        rows.append({"t": text, "mode": mode, **res, "u": UNIDATA})
        sys.stdout.write(
            f"{text!r:32} {mode:16} first: {brief(res['first']):42} again: {brief(res['again'])}\n"
        )
    if out:
        with out.open("w", encoding="utf-8", newline="\n") as f:
            f.writelines(dumps(r) + "\n" for r in rows)
        log(f"wrote {out}: {len(rows)} rows")


# --------------------------------------------------------------------------


def main() -> None:
    global LANG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--lang", default="eng", help="datatrove language code to record (default: eng)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("golden", help="the committed recording (of --lang)")
    g.add_argument("out", type=Path)
    gs = sub.add_parser("goldens", help="the committed multilingual recordings")
    gs.add_argument("dir", type=Path)
    gs.add_argument("keys", nargs="*", metavar="KEY", help="golden keys (default: all but indic-*)")
    c = sub.add_parser("corpus", help="real documents (and their C4-style lines)")
    c.add_argument("out", type=Path)
    c.add_argument("--source", choices=sorted(SOURCES), required=True)
    c.add_argument("--limit", type=int)
    f = sub.add_parser("fuzz", help="generated strings")
    f.add_argument("out", type=Path)
    f.add_argument("n", type=int)
    f.add_argument("seed", type=int)
    v = sub.add_parser("unicode", help="strings aimed at Unicode-version differences")
    v.add_argument("out", type=Path)
    v.add_argument("n", type=int)
    v.add_argument("seed", type=int)
    r = sub.add_parser("rerecord", help="re-record the texts of an existing recording")
    r.add_argument("inp", type=Path)
    r.add_argument("out", type=Path)
    i = sub.add_parser("isalpha", help="characterise the IS_ALPHA history effect")
    i.add_argument("out", type=Path, nargs="?")
    t = sub.add_parser("langtable", help="FineWeb-2 volume per datatrove tokenizer")
    t.add_argument("out", type=Path)
    t.add_argument("--distribution", type=Path, required=True, help=FW2_DISTRIBUTION)
    p = sub.add_parser("_probe")
    p.add_argument("mode")
    a = ap.parse_args()
    LANG = a.lang
    if LANG != "eng":
        assignment(LANG)  # fail early on an unknown code

    if a.cmd == "golden" and LANG == "eng":
        items = golden_real_items() + list(fuzz_items(GOLDEN_FUZZ, GOLDEN_SEED, golden=True))
        run(items, a.out, a.workers, 16)
    elif a.cmd == "golden":
        golden_lang(a.out, a.workers)
    elif a.cmd == "goldens":
        goldens(a.dir, a.keys, a.workers)
    elif a.cmd == "corpus":
        run(corpus_items(a.source, a.limit, SOURCES[a.source][1]), a.out, a.workers, 8)
    elif a.cmd == "fuzz":
        run(fuzz_items(a.n, a.seed), a.out, a.workers, 256)
    elif a.cmd == "unicode":
        run(unicode_items(a.n, a.seed), a.out, a.workers, 256)
    elif a.cmd == "rerecord":

        def rows() -> Iterator[Item]:
            with a.inp.open(encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    yield row["t"], {k: row[k] for k in ("k", "id") if k in row}

        run(rows(), a.out, a.workers, 64)
    elif a.cmd == "isalpha":
        isalpha(a.out)
    elif a.cmd == "langtable":
        language_table(a.distribution, a.out)
    elif a.cmd == "_probe":
        probe(a.mode)


if __name__ == "__main__":
    main()
