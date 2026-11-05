"""Hypothesis strategies for ftfy differential fuzzing (importable without pytest).

Hypothesis never generates surrogates by default (R2 measured 0 in 3,000 draws), and a candidate with
``fix_surrogates`` disabled passes ftfy's whole golden corpus, so lone surrogates and UTF-16 pairs are drawn
explicitly here. Arrow strings are valid UTF-8, so the pinned parquet corpora cannot contain surrogates either:
generated input is the only place this behaviour is exercised.

    from pyroid_parity import strategies as S
    S.texts()        # one line built from mixed pieces (mojibake, entities, controls, surrogates, ...)
    S.documents()    # texts() and multi-line documents (fix_text works line by line)
    S.configs()      # kwargs over every TextFixerConfig field of ftfy 6.3.1

Invisible and look-alike characters are spelled as ``\\N{...}`` / ``\\x..`` escapes or ``chr()`` on purpose.
"""

from __future__ import annotations

import html.entities
import string
from typing import Any

import ftfy
import ftfy.bad_codecs  # registers the sloppy-windows-125x codecs named in CHARMAP_ENCODINGS
from ftfy import chardata
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

ENCODINGS: tuple[str, ...] = tuple(chardata.CHARMAP_ENCODINGS)
NORMALIZATIONS = ("NFC", "NFD", "NFKC", "NFKD", None)
UNESCAPE_HTML = ("auto", True, False)
CP1252 = "sloppy-windows-1252"


def garble(text: str, encodings: list[str] | tuple[str, ...]) -> str:
    """UTF-8 bytes misread as each legacy charset in turn (1 entry = 1 layer of mojibake).

    ``surrogatepass`` lets surrogates through as CESU-8 style bytes (``\\xed\\xa0...``), which ftfy decodes with
    its ``utf-8-variants`` codec.
    """
    for enc in encodings:
        text = text.encode("utf-8", "surrogatepass").decode(enc, errors="replace")
    return text


def utf16_pair(char: str) -> str:
    """The two surrogate code points UTF-16 would use for an astral character."""
    n = ord(char) - 0x10000
    return chr(0xD800 + (n >> 10)) + chr(0xDC00 + (n & 0x3FF))


# Real words with non-ASCII letters, so garbling them yields the mojibake ftfy is built to undo.
WORDS = (
    "caf\xe9",
    "na\xefve",
    "r\xe9sum\xe9",
    "Stra\xdfe",
    "Z\xfcrich",
    "se\xf1or",
    "jalape\xf1o",
    "\xbd",
    "\xa9",
    "\xc5",
    "\xe9",
    "Čeština",
    "Lietuvių",
    "Tiếng Việt",
    "Ελληνικά",
    "Привет",
    "日本語",
    "中文",
    "한국어",
    "עברית",
    "العربية",
    "हिन्दी",
    "ไทย",
    "\N{EURO SIGN}100",
    "\N{EM DASH}",
    "\N{HORIZONTAL ELLIPSIS}",
    "\N{LATIN SMALL LIGATURE FI}ne",
    "\N{LEFT DOUBLE QUOTATION MARK}quoted\N{RIGHT DOUBLE QUOTATION MARK}",
    "\N{LEFT SINGLE QUOTATION MARK}single\N{RIGHT SINGLE QUOTATION MARK}",
    "\N{GRINNING FACE}",
    "\N{THUMBS UP SIGN}\N{EMOJI MODIFIER FITZPATRICK TYPE-4}",
    "\N{MAN}\N{ZERO WIDTH JOINER}\N{WOMAN}\N{ZERO WIDTH JOINER}\N{GIRL}",
    "\N{REGIONAL INDICATOR SYMBOL LETTER F}\N{REGIONAL INDICATOR SYMBOL LETTER R}",
)
# Script blocks for random words: Latin-1/Extended, Greek, Cyrillic, Hebrew, Arabic, Devanagari, Thai, Hangul
# Jamo, general punctuation, CJK, Hangul syllables, emoji.
SCRIPT_RANGES = (
    (0x00C0, 0x024F),
    (0x0370, 0x03FF),
    (0x0400, 0x04FF),
    (0x05D0, 0x05EA),
    (0x0600, 0x06FF),
    (0x0900, 0x097F),
    (0x0E00, 0x0E7F),
    (0x1100, 0x11FF),
    (0x2010, 0x205E),
    (0x4E00, 0x4FFF),
    (0xAC00, 0xAD00),
    (0x1F300, 0x1FAFF),
)
# Single characters and short sequences that individual ftfy fixers react to.
SPECIALS = (
    # line breaks (fix_line_breaks) and segment boundaries: LF, CRLF, CR, NEL, LS, PS, VT, FF
    "\n",
    "\r\n",
    "\r",
    "\x85",
    "\N{LINE SEPARATOR}",
    "\N{PARAGRAPH SEPARATOR}",
    "\x0b",
    "\x0c",
    "\n\n",
    # spaces and invisible characters (restore_byte_a0, remove_control_chars)
    "\xa0",
    "\N{NARROW NO-BREAK SPACE}",
    "\N{FIGURE SPACE}",
    "\N{IDEOGRAPHIC SPACE}",
    "\xad",
    "\N{ZERO WIDTH NO-BREAK SPACE}",
    "\N{ZERO WIDTH SPACE}",
    "\N{ZERO WIDTH JOINER}",
    "\N{LEFT-TO-RIGHT MARK}",
    "\N{LEFT-TO-RIGHT ISOLATE}",
    "\N{INTERLINEAR ANNOTATION ANCHOR}",
    "\x00",
    "\x7f",
    # cp1252 bytes misread as C1 controls (fix_c1_controls) and curly quotes (uncurl_quotes)
    "\x92",
    "\x91\x92\x93\x94",
    "\x80",
    "\x96",
    "\x81",
    "\x9d",
    "\xab",
    "\xbb",
    "\xb4",
    "`",
    "'",
    '"',
    "\N{LEFT DOUBLE QUOTATION MARK}",
    "\N{RIGHT DOUBLE QUOTATION MARK}",
    "\N{LEFT SINGLE QUOTATION MARK}",
    "\N{RIGHT SINGLE QUOTATION MARK}",
    "\N{SINGLE LOW-9 QUOTATION MARK}",
    "\N{DOUBLE LOW-9 QUOTATION MARK}",
    "\N{SINGLE LEFT-POINTING ANGLE QUOTATION MARK}",
    "\N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK}",
    "\N{PRIME}",
    "\N{DOUBLE PRIME}",
    # canned mojibake: 1 and 2 layers of cp1252, "a-grave + NBSP" with the NBSP lost (restore_byte_a0),
    # modified-UTF-8 NUL (C0 80), Cyrillic read as cp1252, lossy U+FFFD
    garble("\N{RIGHT SINGLE QUOTATION MARK}", [CP1252]),
    garble("\N{RIGHT SINGLE QUOTATION MARK}", [CP1252] * 2),
    garble("\N{LEFT DOUBLE QUOTATION MARK}", [CP1252]),
    garble("\N{RIGHT DOUBLE QUOTATION MARK}", [CP1252]),
    garble("\xe9", [CP1252]),
    garble("\xe0", [CP1252]),
    "\xc3 ",
    b"\xc0\x80".decode(CP1252),
    garble("при", [CP1252]),
    garble("\xe9", ["latin-1", CP1252]),
    "\N{REPLACEMENT CHARACTER}",
    "\N{REPLACEMENT CHARACTER}?",
    # HTML context: '<' turns unescape_html="auto" off for that line and every later one
    "<",
    ">",
    "&",
    "<br>",
    "<p>",
    "&lt;",
    "&amp;",
    # normalization and width
    "e\N{COMBINING ACUTE ACCENT}",
    "\N{ANGSTROM SIGN}",
    "\N{ROMAN NUMERAL TWELVE}",
    "\N{CIRCLED DIGIT ONE}",
    "\N{HANGUL CHOSEONG KIYEOK}\N{HANGUL JUNGSEONG A}\N{HANGUL JONGSEONG KIYEOK}",
    "\N{HALFWIDTH KATAKANA LETTER KA}\N{HALFWIDTH KATAKANA VOICED SOUND MARK}",
    "\N{HALFWIDTH KATAKANA SEMI-VOICED SOUND MARK}",
    "\N{TAG LATIN CAPITAL LETTER A}",
    chr(0xFFFE),
    chr(0xFFFF),
    chr(0x10FFFF),
)

# ------------------------------------------------------------------------------------------------- pieces

lone_surrogates: SearchStrategy[str] = st.characters(categories=("Cs",))
astral: SearchStrategy[str] = st.characters(min_codepoint=0x10000)
surrogate_pairs: SearchStrategy[str] = astral.map(utf16_pair)
# Runs of surrogates: reversed pairs (low, high), doubled highs, pairs next to lone halves.
surrogate_runs: SearchStrategy[str] = st.lists(
    st.one_of(
        lone_surrogates,
        surrogate_pairs,
        st.sampled_from([chr(0xD83D), chr(0xDE00), chr(0xDBFF) + chr(0xDFFF), chr(0xD800) + chr(0xDC00)]),
    ),
    min_size=1,
    max_size=4,
).map("".join)

script_words: SearchStrategy[str] = st.text(
    st.one_of(
        [st.characters(min_codepoint=a, max_codepoint=b) for a, b in SCRIPT_RANGES]
        + [st.sampled_from(" .,'")]
    ),
    min_size=1,
    max_size=12,
)
words: SearchStrategy[str] = st.one_of(
    st.sampled_from(WORDS),
    script_words,
    st.text(st.characters(categories=("L", "M", "N", "P", "S", "Zs")), min_size=1, max_size=12),
)

_LOSSY = str.maketrans(dict.fromkeys(range(0x80, 0xA0), "\N{REPLACEMENT CHARACTER}"))
mojibake: SearchStrategy[str] = st.builds(
    garble,
    st.one_of(words, st.lists(words, min_size=1, max_size=4).map(" ".join)),
    st.lists(st.sampled_from(ENCODINGS), min_size=1, max_size=2),
).flatmap(lambda s: st.sampled_from([s, s.replace("\xa0", " "), s.translate(_LOSSY)]))  # NBSP lost / lossy
# CESU-8 surrogate pairs (emoji as Java / MySQL "utf8" write them) misread as latin-1 or cp1252.
cesu8_mojibake: SearchStrategy[str] = st.builds(
    garble,
    surrogate_pairs,
    st.lists(st.sampled_from(["latin-1", CP1252]), min_size=1, max_size=1),
)

_ENTITY_NAMES = sorted({*chardata.HTML_ENTITIES, *("&" + k for k in html.entities.html5)})
html_entities: SearchStrategy[str] = st.one_of(
    st.sampled_from(_ENTITY_NAMES),  # named (incl. ftfy's all-caps "&EACUTE;" forms), with and without ';'
    st.integers(0, 0x110010).map(lambda n: f"&#{n};"),
    st.builds(
        lambda n, x, up: f"&#{x}{n:X};" if up else f"&#{x}{n:x};",
        st.integers(0, 0x110010),
        st.sampled_from("xX"),
        st.booleans(),
    ),
    st.sampled_from(
        [
            "&#xD800;",
            "&#xDFFF;",
            "&#0;",
            "&#128;",
            "&#150;",
            "&#x110000;",
            "&#;",
            "&nosuch;",
            "&AMP;",
            "&amp",
            "&ampx;",
            "&#x1F600;",
            "&#128512;",
            "&EACUTE;",
            "&NTILDE;",
            "&Eacute;",
            "&eacute",
        ]
    ),
)
ansi_escapes: SearchStrategy[str] = st.one_of(
    st.builds(
        lambda params, letter: f"\x1b[{params}{letter}",
        st.from_regex(r"[0-9;]{0,8}", fullmatch=True),
        st.sampled_from(string.ascii_letters),
    ),
    st.sampled_from(
        ["\x1b[0m", "\x1b[31m", "\x1b[1;32;40m", "\x1b[K", "\x1b]0;title\x07", "\x1b(B", "\x1b", "\x1b["]
    ),
)
ligatures: SearchStrategy[str] = st.sampled_from(sorted(chr(c) for c in chardata.LIGATURES))
fullwidth: SearchStrategy[str] = st.one_of(
    st.sampled_from(sorted(chr(c) for c in chardata.WIDTH_MAP)),
    st.characters(min_codepoint=0xFF00, max_codepoint=0xFFEF),
)
controls: SearchStrategy[str] = st.one_of(
    st.characters(categories=("Cc",)),  # C0, DEL, C1 (incl. NUL and NEL)
    st.characters(categories=("Cf",)),  # BOM, ZW(N)J, bidi controls, interlinear annotation, tags
    st.sampled_from(sorted(chr(c) for c in chardata.CONTROL_CHARS)),
)
specials: SearchStrategy[str] = st.sampled_from(SPECIALS)
random_text: SearchStrategy[str] = st.one_of(
    st.text(max_size=12),  # any non-surrogate code point, including unassigned ones
    st.text(st.characters(codec="ascii"), max_size=24),
)

pieces: SearchStrategy[str] = st.one_of(
    words,
    mojibake,
    cesu8_mojibake,
    html_entities,
    ansi_escapes,
    ligatures,
    fullwidth,
    controls,
    specials,
    lone_surrogates,
    surrogate_pairs,
    surrogate_runs,
    random_text,
)


# ------------------------------------------------------------------------------------------------ texts


def texts(max_pieces: int = 12) -> SearchStrategy[str]:
    """One string made of up to ``max_pieces`` pieces; sometimes the whole string is garbled once more."""
    line = st.lists(pieces, max_size=max_pieces).map("".join)
    return st.one_of(
        line, st.builds(garble, line, st.lists(st.sampled_from(ENCODINGS), min_size=1, max_size=1))
    )


def documents(max_lines: int = 5) -> SearchStrategy[str]:
    """``texts()`` plus multi-line documents (fix_text fixes each line as its own segment)."""
    return st.one_of(texts(), st.lists(texts(6), min_size=2, max_size=max_lines).map("\n".join))


# ---------------------------------------------------------------------------------------------- configs

FIELD_STRATEGIES: dict[str, SearchStrategy[Any]] = {
    name: st.booleans()
    for name, default in ftfy.TextFixerConfig._field_defaults.items()
    if isinstance(default, bool)
} | {
    "unescape_html": st.sampled_from(UNESCAPE_HTML),
    "normalization": st.sampled_from(NORMALIZATIONS),
    # 0 and negatives make ftfy 6.3.1's fix_text loop forever, so they are not generated.
    "max_decode_length": st.one_of(st.integers(1, 64), st.just(1_000_000)),
}
if _changed := set(ftfy.TextFixerConfig._fields) ^ set(FIELD_STRATEGIES):  # refuse to silently under-fuzz
    raise ImportError(
        f"pyroid_parity.strategies: TextFixerConfig fields differ from ftfy 6.3.1: {sorted(_changed)}"
    )


def configs() -> SearchStrategy[dict[str, Any]]:
    """Keyword arguments for ``fix_text`` / ``fix_encoding``: any subset of TextFixerConfig fields (often none)."""
    return st.fixed_dictionaries({}, optional=FIELD_STRATEGIES)
