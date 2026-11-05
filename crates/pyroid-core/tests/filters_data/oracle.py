"""Record what datatrove 0.10.1 returns for English word and sentence splitting.

Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace, and spaCy 3.8.18
(MIT), © ExplosionAI GmbH; modified.

datatrove's `split_into_words` / `split_into_sentences` (default language
"eng") run spaCy 3.8.18 `spacy.blank("en")` plus the `sentencizer` pipe. The
Rust port (`pyroid_core::filters::tokenizer`) must give identical results; this
script records the oracle and `tokenizer_test.rs` replays it.

Run inside the pyroid-dev container (never on the host), with a venv holding
datatrove==0.10.1, spacy==3.8.18, regex and pyarrow:

    python oracle.py golden OUT.jsonl                # the committed recording
    python oracle.py corpus OUT.jsonl --source dev|100k|adversarial|fineweb2 [--limit N]
    python oracle.py fuzz OUT.jsonl N SEED           # N generated strings
    python oracle.py unicode OUT.jsonl N SEED        # N strings aimed at Unicode versions
    python oracle.py rerecord IN.jsonl OUT.jsonl     # IN's texts, on this interpreter
    python oracle.py isalpha [OUT.jsonl]             # the IS_ALPHA history effect

Each output line is one text, recorded with one call of each datatrove function:

    {"t": text, "w": split_into_words(text), "s": len(split_into_sentences(text)),
     "u": unicodedata.unidata_version}

or, when datatrove raises, {"t": ..., "raise": "ExceptionType", "at": "w"|"s", "u": ...}.
Rows may also carry "k" (what the text is: "doc", "line", "excerpt" or the
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
import functools
import glob
import heapq
import itertools
import json
import multiprocessing
import random
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

UNIDATA = unicodedata.unidata_version
SOURCES: dict[str, tuple[list[str], bool]] = {
    # name: (parquet files, also record C4-style lines)
    "dev": (["/data/fineweb/fineweb-dev-10k.parquet"], True),
    "100k": (["/data/fineweb/fineweb-100k.parquet"], False),
    "adversarial": (["/data/parity/adversarial.parquet"], True),
    "fineweb2": (sorted(glob.glob("/data/fineweb2/*.parquet")), False),
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
    """One oracle row: datatrove's words and sentence count for `text`."""
    from datatrove.utils.text import split_into_sentences, split_into_words

    row: dict[str, Any] = {"t": text}
    at = "w"
    try:
        words = split_into_words(text)
        at = "s"
        n = len(split_into_sentences(text))
    except Exception as e:
        row["raise"] = type(e).__name__
        row["at"] = at
    else:
        row["w"] = list(words)
        row["s"] = n
    row["u"] = UNIDATA
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


def run(items: Iterable[Item], path: Path, workers: int, chunksize: int) -> None:
    """Record every item with a fork pool, in input order, streaming to `path`."""
    t0 = time.monotonic()
    record("Warm up. Load spaCy before forking.")
    raises: Counter[str] = Counter()
    n = 0
    ctx = multiprocessing.get_context("fork")
    # Fork before reading any parquet, so no pyarrow threads exist at fork time.
    with ctx.Pool(workers) as pool, path.open("w", encoding="utf-8", newline="\n") as out:
        for block in _blocks(items, 20_000):
            for line, raised in pool.imap(_job, block, chunksize):
                out.write(line + "\n")
                n += 1
                if raised:
                    raises[raised] += 1
    dt = time.monotonic() - t0
    log(
        f"wrote {path}: {n} rows, Unicode {UNIDATA}, Python {sys.version.split()[0]}, "
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
                yield text, f"{stem}:{doc_id}" if source == "fineweb2" else str(doc_id)


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
        self.pieces: list[tuple[Callable[[], str], float]] = [
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
        self.texts: list[tuple[Callable[[], str], float]] = [
            (self.t_space, 4),
            (self.t_punct, 3),
            (self.t_single, 12),
            (self.t_glued, 15),
            (self.t_mixed, 34),
            (self.t_english, 20),
            (self.t_sentences, 8),
            (self.t_boundary, 10),
            (self.t_long, 0.15 if golden else 1.5),
            (self.t_soup, 0.7),
        ]
        self.piece_fns = [fn for fn, _ in self.pieces]
        self.piece_cum = list(itertools.accumulate(w for _, w in self.pieces))
        self.text_fns = [fn for fn, _ in self.texts]
        self.text_cum = list(itertools.accumulate(w for _, w in self.texts))

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
    f = Fuzzer(seed, golden)
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


def greedy(cands: list[tuple[str, frozenset[str]]], k: int, covered: set[str]) -> list[int]:
    """Lazy greedy set cover: up to `k` candidates adding the most new features per length."""

    def score(i: int) -> float:
        text, feats = cands[i]
        return len(feats - covered) / (1 + len(text) / 300)

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
            [sys.executable, "-I", str(Path(__file__).resolve()), "_probe", mode],
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("golden", help="the committed recording")
    g.add_argument("out", type=Path)
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
    p = sub.add_parser("_probe")
    p.add_argument("mode")
    a = ap.parse_args()

    if a.cmd == "golden":
        items = golden_real_items() + list(fuzz_items(GOLDEN_FUZZ, GOLDEN_SEED, golden=True))
        run(items, a.out, a.workers, 16)
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
    elif a.cmd == "_probe":
        probe(a.mode)


if __name__ == "__main__":
    main()
