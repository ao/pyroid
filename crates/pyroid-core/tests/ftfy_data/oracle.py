"""Record what the real ftfy 6.3.1 does, for the Rust differential tests.

Derived from ftfy 6.3.1 (Apache-2.0), © Robyn Speer; modified.

Run inside the pyroid-dev container with an interpreter that has
`ftfy==6.3.1` installed (never on the host):

    python oracle.py golden OUT.jsonl          # ftfy's own test inputs
    python oracle.py fuzz OUT.jsonl N SEED     # N random inputs
    python oracle.py mojibake OUT.jsonl N SEED # N realistic mojibake inputs
    python oracle.py corpus OUT.jsonl A.parquet [B.parquet ...]  # fix_text on real docs

Each output line is one input and the calls made on it: the function, its
keyword arguments and ftfy's result. Strings with lone surrogates are stored as
{"h": hex of the "surrogatepass" UTF-8 encoding}, so they survive JSON.
`tests/ftfy_oracle.rs` replays the file against `pyroid_core::ftfy`.
"""

from __future__ import annotations

import json
import random
import sys
import unicodedata
from pathlib import Path

import ftfy
import ftfy.bad_codecs  # noqa: F401  (registers the sloppy codecs)
from ftfy.chardata import CHARMAP_ENCODINGS

HERE = Path(__file__).resolve().parent
FLAGS = [
    name
    for name, default in ftfy.TextFixerConfig._field_defaults.items()
    if isinstance(default, bool) and name != "explain"
]


def enc(s: str):
    """A JSON string, or {"h": hex} when lone surrogates make UTF-8 impossible."""
    try:
        s.encode("utf-8")
        return s
    except UnicodeEncodeError:
        return {"h": s.encode("utf-8", "surrogatepass").hex()}


def call(fn: str, text: str, kwargs: dict, explain_cfg: bool | None = None) -> dict:
    """Call ftfy; `explain_cfg` passes `config=TextFixerConfig(explain=...)`."""
    call_kwargs = dict(kwargs)
    if explain_cfg is not None:
        call_kwargs = {"config": ftfy.TextFixerConfig(explain=explain_cfg), **kwargs}
    row: dict = {"fn": fn}
    if kwargs:
        row["kwargs"] = kwargs
    if explain_cfg is not None:
        row["config_explain"] = explain_cfg
    try:
        res = getattr(ftfy, fn)(text, **call_kwargs)
    except Exception as e:  # noqa: BLE001  (the exception type is part of parity)
        row["error"] = type(e).__name__
    else:
        if isinstance(res, tuple):
            row["out"] = enc(res[0])
            row["explanation"] = None if res[1] is None else [list(s) for s in res[1]]
        else:
            row["out"] = enc(res)
    return row


def write(out, text: str, calls: list[dict]) -> None:
    out.write(json.dumps({"in": enc(text), "calls": calls}, ensure_ascii=False) + "\n")


def standard_calls(text: str) -> list[dict]:
    only_encoding = {f: False for f in FLAGS if f not in (
        "fix_encoding", "restore_byte_a0", "replace_lossy_sequences",
        "decode_inconsistent_utf8", "fix_c1_controls")}
    return [
        call("fix_text", text, {}),
        call("fix_text", text, {}, explain_cfg=True),
        call("fix_text_segment", text, {}),
        call("fix_and_explain", text, {}),
        call("fix_and_explain", text, {}, explain_cfg=False),
        call("fix_encoding", text, {}),
        call("fix_encoding_and_explain", text, {}),
        call("fix_text", text, {**only_encoding, "unescape_html": False, "normalization": None}),
    ]


# ---------------------------------------------------------------------------
# golden: ftfy's own test inputs
# ---------------------------------------------------------------------------

DOCTEST_INPUTS = [
    "âœ” No problems",
    "&macr;\\_(ã\x83\x84)_/&macr;",
    "Broken text&hellip; it&#x2019;s ﬂubberiﬁc!",
    "ＬＯＵＤ　ＮＯＩＳＥＳ",
    "sÃ³",
    "voilÃ le travail",
    "Ã³",
    "&ATILDE;&SUP3;",
    "schÃ¶n",
    "&lt;tag&gt;",
    "&Jscr;ohn &HilbertSpace;ancock",
    "&checkmark;",
    "P&eacute;rez",
    "P&EACUTE;REZ",
    "BUNDESSTRA&SZLIG;E",
    "&ntilde; &Ntilde; &NTILDE; &nTILDE;",
    "\033[36;44mI'm blue, da ba dee da ba doo...\033[0m",
    "“here’s a test”",
    "ﬂuﬃeﬆ",
    "Ｕﾀｰﾝ",
    "This string is made of two things: " "1. Unicode " "2. Spite",
    "Content-type: text/plain\r\n\r\nHi.",
    "This is how Microsoft \r trolls Mac users",
    "What is this \x85 I don't even",
    "💩",
    "\udca9\ud83d",
    chr(0xDBFF) + chr(0xDFFF),
    chr(0xD800) + chr(0xDC00),
    "ï»¿",
    "﻿Sometimes, ￼bad ideas \x7f￺like these characters￻ "
    "⁪get standardized.\r\n",
    "This flag has a dragon on it 🏴󠁧󠁢󠁷󠁬󠁳󠁿",
    "#superman #ohio 🏴\U000e0075\U000e0073\U000e006f\U000e0068\U000e007f #cleveland #usa 🇺🇸",
    "\001\033[36;44mfoo",
    "&amp;\n<html>\n&amp;",
    "&lt;&gt;",
    "jednocze&sacute;nie",
    "JEDNOCZE&Sacute;NIE",
    "ellipsis&#133;",
    "ellipsis&#x85;",
    "broken&#x81;",
    "&amp;amp;amp;",
    "euro &#x80;",
    "EURO &EURO;",
    "not an entity &#20x6;",
    "V&SCARON;ICHNI",
    "&#xffff;",
    "&#xffffffff;",
    "this is just informal english &not html",
    "&#59; &#x3b; &#0; &#1; &#13; &#x110000; &#55296;",
    "Ã©À\n",
    "x\xed\xa0\xbd\xed\xb8\n",
]


def golden(path: str) -> None:
    cases = json.loads((HERE / "test_cases.json").read_text("utf-8"))
    texts = []
    for case in cases:
        texts.append(case["original"])
        texts.append(case["original"].encode("utf-8").decode("latin-1"))
    texts += DOCTEST_INPUTS
    with open(path, "w", encoding="utf-8") as out:
        for t in texts:
            write(out, t, standard_calls(t))
        for t in ("&amp;\n<html>\n&amp;", "&lt;&gt;"):
            write(out, t, [call(fn, t, {"unescape_html": ue})
                           for ue in (True, False, "auto")
                           for fn in ("fix_text", "fix_text_segment")])
        for t in ("ellipsis&#133;", "ellipsis&#x85;"):
            write(out, t, [call("fix_text_segment", t, {"normalization": "NFKC"})])
    print(f"wrote {path}: {len(texts)} inputs, Unicode {unicodedata.unidata_version}")


# ---------------------------------------------------------------------------
# fuzz: random inputs aimed at ftfy's corners
# ---------------------------------------------------------------------------

# Characters assigned in Unicode 15.0..17.0, with marks, to exercise the
# per-version tables (normalization, \w, \d).
NEW_CHARS = (
    [chr(c) for c in range(0x11F00, 0x11F5A)]  # Kawi (15.0): marks and digits
    + [chr(c) for c in range(0x1E4D0, 0x1E4FA)]  # Nag Mundari (15.0)
    + [chr(c) for c in range(0x2EBF0, 0x2EBF8)]  # CJK ext I (15.1)
    + [chr(c) for c in range(0x1E5D0, 0x1E5FC)]  # Ol Onal (16.0)
    + [chr(c) for c in range(0x16D40, 0x16D7A)]  # Kirat Rai (16.0)
    + [chr(c) for c in range(0x11380, 0x113E3)]  # Tulu-Tigalari (16.0)
    + [chr(c) for c in range(0x116D0, 0x116E4)]  # Myanmar ext C (16.0)
    + [chr(c) for c in range(0x11DB0, 0x11DEA)]  # (17.0 additions)
    + [chr(c) for c in range(0x1E6C0, 0x1E700)]  # (17.0 additions)
    + [chr(c) for c in range(0x10940, 0x10960)]  # (17.0 additions)
)
SPECIALS = [
    "&amp;", "&eacute;", "&EACUTE;", "&#x1F600;", "&#59;", "&#0;", "&#x110000;", "&lt;",
    "&nbsp;", "&amp;amp;", "&#X41;", "&#x;", "&#;", "&NTILDE;", "&nTILDE;", "&ETH;", "&SZLIG;",
    "&" + "a" * 25 + ";", "&CounterClockwiseContourIntegral;", "&#000000000000000000065;",
    "\x1b[31m", "\x1b[1;2;3m", "\x1b[", "\x1b[١٢m", "\x1b[0", "\x1b\x1b[m",
    "“", "”", "‘", "’", "ʼ", "‛", "‟", "ﬁ", "ﬀ", "Ĳ", "ǅ", "ＡＢＣ", "　", "ｶﾞ", "￣",
    "\r\n", "\r", " ", " ", "\x85", "\x92", "\x80", "\x81", "\x9d", "﻿",
    "\x00", "\x1a", "\x7f", "\x0b", "\xa0", "⁪", "￼", "�", "Ã ", "Ã", " ",
    "<", "\n", "\t", "à", "Â", "Ã©", "â€™", "â€", "Ð", "Ã\xa0", "Ã la", "fÃ cil", "Ã s ",
    "̀", "́", "̧", "ͅ", "ཱི", "가", "각",
    "é", "Å", "Ω", "ﬃ", "²", "½", "µ", "Œ", "œ", "×", "√", "≈", "‚Ä",
    "вЂ", "ГўВЂВ", "β€", "ā€", "Ø§", "Ù„", "", "\U0001f600", "\U000e0067",
]


def pool_chars(rng: random.Random) -> str:
    kind = rng.randrange(10)
    if kind == 0:
        return chr(rng.randrange(0x20, 0x7F))
    if kind == 1:
        return chr(rng.randrange(0x80, 0x250))
    if kind == 2:
        return chr(rng.randrange(0x370, 0x530))
    if kind == 3:
        return chr(rng.randrange(0x2000, 0x2600))
    if kind == 4:
        enc = rng.choice(CHARMAP_ENCODINGS)
        return bytes([rng.randrange(0x80, 0x100)]).decode(enc, errors="replace")
    if kind == 5:
        return rng.choice(NEW_CHARS)
    if kind == 6:
        return chr(rng.randrange(0xD800, 0xE000))  # lone surrogate
    if kind == 7:
        return chr(rng.choice([rng.randrange(0x300, 0x370), rng.randrange(0x1AB0, 0x1B00),
                               rng.randrange(0x20D0, 0x2100), rng.randrange(0xFE20, 0xFE30)]))
    if kind == 8:
        cp = rng.randrange(0, 0x110000)
        return chr(cp)
    return chr(rng.randrange(0x4E00, 0x4F00))


def word(rng: random.Random) -> str:
    alphabet = rng.choice([
        "abcdefghijklmnopqrstuvwxyz ",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ .,?!",
        "àáâãäåæçèéêëìíîïñòóôõöøùúûüýÿßœ ",
        "αβγδεζηθικλμνξοπρστυφχψω ΑΒΓΔΕ",
        "абвгдежзийклмнопрстуфхцчшщъыьэюя ВГРС",
        "日本語中文한국어 ",
        "0123456789-—–'\"…•€£¥©®™°±",
    ])
    return "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 12)))


def garble(rng: random.Random, s: str) -> str:
    for _ in range(rng.randrange(1, 3)):
        enc = rng.choice(CHARMAP_ENCODINGS + ["windows-1252", "windows-1251"])
        s = s.encode("utf-8", "surrogatepass").decode(enc, errors=rng.choice(["replace", "ignore"]))
    if rng.random() < 0.2:
        s = s.replace("\xa0", " ")
    return s


def piece(rng: random.Random) -> str:
    kind = rng.randrange(8)
    if kind == 0:
        return word(rng)
    if kind == 1:
        return garble(rng, word(rng) + rng.choice(["", "’", "“", "—", "é", "😀", "ü"]))
    if kind == 2:
        return rng.choice(SPECIALS)
    if kind == 3:
        return "".join(pool_chars(rng) for _ in range(rng.randrange(1, 6)))
    if kind == 4:
        hi, lo = chr(rng.randrange(0xD800, 0xDC00)), chr(rng.randrange(0xDC00, 0xE000))
        return rng.choice([hi + lo, lo + hi, hi, lo, hi + hi + lo])
    if kind == 5:
        # embed mojibake inside correct text
        return word(rng) + garble(rng, rng.choice(["é", "’", "€", "日本", "ñ", "—"])) + word(rng)
    if kind == 6:
        enc = rng.choice(CHARMAP_ENCODINGS)
        raw = bytes(rng.randrange(0x80, 0x100) for _ in range(rng.randrange(1, 5)))
        return raw.decode(enc, errors="replace")
    return " "


def random_text(rng: random.Random) -> str:
    n = rng.choice([0, 1, 2, 3, 5, 8, 12, 20])
    s = "".join(piece(rng) for _ in range(n))
    if rng.random() < 0.02:
        s = s * rng.randrange(2, 40)
    return s


def random_kwargs(rng: random.Random) -> dict:
    kw: dict = {}
    for f in FLAGS:
        if rng.random() < 0.15:
            kw[f] = rng.random() < 0.5
    if rng.random() < 0.2:
        kw["unescape_html"] = rng.choice(["auto", True, False])
    if rng.random() < 0.2:
        kw["normalization"] = rng.choice(["NFC", "NFD", "NFKC", "NFKD", None])
    if rng.random() < 0.1:
        kw["max_decode_length"] = rng.randrange(1, 40)
    if rng.random() < 0.2:
        kw["explain"] = rng.random() < 0.5
    return kw


def fuzz(path: str, n: int, seed: int) -> None:
    rng = random.Random(seed)
    fns = ["fix_text", "fix_text", "fix_text", "fix_and_explain", "fix_encoding",
           "fix_encoding_and_explain", "fix_text_segment"]
    with open(path, "w", encoding="utf-8") as out:
        for _ in range(n):
            text = random_text(rng)
            fn = rng.choice(fns)
            kw = random_kwargs(rng) if rng.random() < 0.6 else {}
            write(out, text, [call(fn, text, kw)])
    print(f"wrote {path}: {n} calls, seed {seed}, Unicode {unicodedata.unidata_version}")


# ---------------------------------------------------------------------------
# mojibake: realistic text pushed through the mistakes ftfy is built to undo
# ---------------------------------------------------------------------------

SAMPLES = [
    "Le café est très bon, à la carte — voilà !",
    "Ça m’a coûté 20 € l’été dernier… «incroyable»",
    "Größe: 5 m², Straße “Müller” – geöffnet",
    "¿Dónde está la estación? ¡Mañana será!",
    "Это пример текста на русском языке — «цитата»",
    "Ελληνικά: Καλημέρα κόσμε, τι κάνεις;",
    "Zażółć gęślą jaźń. Příliš žluťoučký kůň úpěl ďábelské ódy.",
    "Türkçe: Işık, ağaç, şemsiye, ğ ç ö ü",
    "Lietuvių kalba: ąčęėįšųūž ĄČĘĖĮŠŲŪŽ",
    "日本語のテキストと中文文本，还有한국어 문장",
    "Emoji: 😀 👍🏽 🇺🇸 ❤️ — and “quotes”",
    "Price: £5 · ¥100 · ¢50 · ±2 · ½ · ° · µ · ×",
    "Árvíztűrő tükörfúrógép ŐŰ őű",
    "à la mode, às vezes, àquele, fà cil",
    "Ølen og ærøskøbing på Åland",
    "ﬁne ﬂour, Ｗｉｄｅ, ｶﾀｶﾅ",
    "bullet • dagger † ‰ ™ © ® ¶ §",
    "\u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0647\u0646\u0627",
]


def mojibake_text(rng: random.Random) -> str:
    s = " ".join(rng.choice(SAMPLES) for _ in range(rng.randrange(1, 4)))
    if rng.random() < 0.3:
        words = s.split(" ")
        k = rng.randrange(len(words))
        words[k] = "\x00" if rng.random() < 0.1 else words[k]
        s = " ".join(words)
    mode = rng.randrange(6)
    if mode == 0:  # plain UTF-8 misread once or twice
        b = s.encode("utf-8")
        t = garble_bytes(rng, b)
    elif mode == 1:  # CESU-8 / Java nulls
        b = s.encode("utf-16-be", "surrogatepass")
        units = [b[i:i + 2] for i in range(0, len(b), 2)]
        out = bytearray()
        for u in units:
            cp = int.from_bytes(u, "big")
            out += b"\xc0\x80" if cp == 0 else chr(cp).encode("utf-8", "surrogatepass")
        t = garble_bytes(rng, bytes(out))
    elif mode == 2:  # only part of the text is mojibake
        cut = rng.randrange(len(s) + 1)
        t = s[:cut] + garble_bytes(rng, s[cut:].encode("utf-8"))
    elif mode == 3:  # mojibake of mojibake, then fixed halfway
        t = garble_bytes(rng, garble_bytes(rng, s.encode("utf-8")).encode("utf-8"))
    elif mode == 4:  # lossy: undefined bytes became U+FFFD or "?"
        enc = rng.choice(["windows-1252", "windows-1251", "windows-1250", "windows-1253"])
        t = s.encode("utf-8").decode(enc, errors="replace")
        if rng.random() < 0.5:
            t = t.replace("\ufffd", "?")
    else:
        t = s
    if rng.random() < 0.3:
        t = t.replace("\xa0", " ")
    if rng.random() < 0.2:
        t = t + rng.choice(["\n", "\r\n", "\n\n", " \n"])
    if rng.random() < 0.1:
        t = rng.choice(SPECIALS) + t
    return t


def garble_bytes(rng: random.Random, b: bytes) -> str:
    enc = rng.choice(CHARMAP_ENCODINGS + ["windows-1252", "cp850", "koi8-r", "iso-8859-15"])
    return b.decode(enc, errors=rng.choice(["replace", "replace", "ignore"]))


def mojibake(path: str, n: int, seed: int) -> None:
    rng = random.Random(seed)
    with open(path, "w", encoding="utf-8") as out:
        for _ in range(n):
            text = mojibake_text(rng)
            calls = [call("fix_text", text, {}), call("fix_and_explain", text, {})]
            if rng.random() < 0.3:
                calls.append(call("fix_text", text, random_kwargs(rng)))
            write(out, text, calls)
    print(f"wrote {path}: {n} inputs, seed {seed}, Unicode {unicodedata.unidata_version}")


def corpus(path: str, files: list[str]) -> None:
    import pyarrow.parquet as pq

    n = changed = 0
    with open(path, "w", encoding="utf-8") as out:
        for f in files:
            for text in pq.read_table(f, columns=["text"]).column("text").to_pylist():
                fixed = ftfy.fix_text(text)
                c: dict = {"fn": "fix_text"}
                if fixed == text:
                    c["same"] = True
                else:
                    c["out"] = enc(fixed)
                    changed += 1
                write(out, text, [c])
                n += 1
    print(f"wrote {path}: {n} docs ({changed} changed), Unicode {unicodedata.unidata_version}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "golden":
        golden(sys.argv[2])
    elif mode == "fuzz":
        fuzz(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif mode == "mojibake":
        mojibake(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif mode == "corpus":
        corpus(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit(__doc__)
