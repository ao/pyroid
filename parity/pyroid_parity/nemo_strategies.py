"""Hypothesis strategies for the ``nemo`` suite: text that reaches every branch of NeMo's heuristics and modifiers.

Documents are built from fragments NeMo's code looks for (whitespace of every kind, newline runs, '#' runs,
ellipsis and end marks, bullets, common English words, policy strings, URLs with ``%XX``, markdown nesting,
quotes, brackets, digits) plus characters whose ``lower()`` or case folding is not one-to-one (U+0130, U+212A,
U+017F), astral characters and free text; lines and paragraphs are repeated to make duplicate n-grams, lines
and paragraphs. One branch adds lone surrogates (a ``str`` can hold them, a reader's Arrow column cannot).
"""

from __future__ import annotations

import sys

from hypothesis import strategies as st

#: Every ``str.isspace()`` character, plus U+0008, which NeMo's WhiteSpaceFilter counts and str.split does not.
WHITESPACE = sorted({chr(c) for c in range(sys.maxunicode + 1) if chr(c).isspace()} | {"\b"})

#: NeMo's bullet_list, ellipsis_marks and paragraph/URL/markdown edge cases, as escapes where the glyph is ambiguous.
BULLETS = ["\u2022", "\u2023", "\u2043", "\u204c", "\u204d", "\u2219", "\u25cb", "\u25cf", "\u25d8", "\u25e6"]
BULLETS += ["\u29be", "\u29bf"]
UNICODE_EDGES = [
    "\u0130",  # LATIN CAPITAL LETTER I WITH DOT ABOVE: lower() is "i" + U+0307
    "\u212a",  # KELVIN SIGN: lower() is "k"
    "\u017f",  # LATIN SMALL LETTER LONG S: matches "s" under re.IGNORECASE
    "\ufb03",  # LATIN SMALL LIGATURE FFI
    "\u00df",
    "\u03a3",
    "\u01c5",
    "\u00e9",  # e with acute, precomposed
    "e\u0301",  # e + COMBINING ACUTE ACCENT
    "\u65e5\u672c\u8a9e",
    "\u4e2d\u6587\u5206\u8bcd\u6d4b\u8bd5",
    "\u0416",
    "\u2177",  # SMALL ROMAN NUMERAL EIGHT: alphabetic, not a letter
    "\u00aa",
    "\u0663",  # ARABIC-INDIC DIGIT THREE: not [[:digit:]] in NeMo's regex
    "\u00b3",
    "\U0001f600",
    "\U0001d518",
    "\U00010400",
    "\U000e0001",
    "\x00",
    "\x1b[31m",
    "&amp;",
    "\u00c3\u00a9",  # mojibake of e-acute
]

FRAGMENTS = [
    # line and paragraph structure
    "\n",
    "\n\n",
    "\n\n\n",
    "\r\n",
    "\r\n\r\n\r\n",
    " \n ",
    # symbols and marks
    "#",
    "##",
    "###",
    "# #",
    "...",
    "[...]",
    "\u2026",
    "(...)",
    "[\u2026]",
    "-\u00bb",
    "\u00bb",
    "read more..",
    "read more",
    "Read More",
    "READ MORE",
    ".",
    "?",
    "!",
    '"',
    "'",
    *BULLETS,
    "- ",
    "* ",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "\u27e8",
    "\u27e9",
    # words
    "the",
    "be",
    "to",
    "of",
    "and",
    "that",
    "have",
    "with",
    "The",
    "THE",
    "word",
    "a",
    "x",
    "0",
    "123",
    "w" * 1001,
    # boilerplate
    "lorem ipsum",
    "Lorem Ipsum",
    "terms of use",
    "Privacy Policy",
    "cookie policy",
    "uses cookies",
    "privacy & cookies policy",
    "use of cookies",
    "TERMS OF USE",
    # URLs
    "http://",
    "https://",
    "www.",
    "http://example.com/a%20b",
    "https://x.org/%zz%4",
    "http://porn.example/",
    "https://example.com/PORN",
    "HTTPS://WWW.EXAMPLE.COM/x",
    "http\u017f://x",
    "https://a.b/c?d=e&f=g#h~",
    "http://a(b)c,d!*",
    "www.example.org",
    # markdown and quotes
    "**",
    "*",
    "_",
    "***a**",
    "**a*b**",
    "[a](b)(c)",
    "[x](y)",
    "__init__",
    "_a_b_",
    '"a"',
    '""',
    *UNICODE_EDGES,
    "__label__",
    "__label__hq",
]

_piece = st.one_of(
    st.sampled_from(FRAGMENTS),
    st.sampled_from(WHITESPACE),
    st.text(st.characters(exclude_categories=("Cs",)), max_size=6),
)
_line = st.lists(_piece, max_size=12).map("".join)


@st.composite
def _repeated(draw: st.DrawFn) -> str:
    lines = draw(st.lists(_line, min_size=1, max_size=4))
    reps = draw(st.integers(1, 4))
    sep = draw(st.sampled_from(["\n", "\n\n", " ", "\n\n\n", ""]))
    order = draw(st.permutations(range(len(lines) * reps)))
    return sep.join(lines[k % len(lines)] for k in order)


_surrogates = st.text(st.characters(categories=("Cs", "Ll", "Zs", "Nd", "Po")), max_size=20)

#: One document, without lone surrogates.
document = st.one_of(_line, _repeated(), st.lists(_piece, max_size=40).map("".join))
#: One document, lone surrogates allowed in one branch.
document_any = st.one_of(document, _surrogates)
#: A batch of 1-4 documents (str API and batch API compared on the same inputs).
batch = st.lists(document_any, min_size=1, max_size=4)
batch_no_surrogates = st.lists(document, min_size=1, max_size=4)
