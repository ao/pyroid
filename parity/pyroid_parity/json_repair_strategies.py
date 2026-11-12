"""Hypothesis strategies for json_repair differential fuzzing: JSON-ish text from a token grammar.

    from pyroid_parity import json_repair_strategies as S
    S.json_ish()      # tokens: structure, every quote kind, escapes, literals, odd numbers, comments, fences,
                      # parentheses, unicode whitespace, letters and digits of every script, lone surrogates
    S.near_json()     # valid JSON (json.dumps of generated values) cut, spliced or wrapped in prose
    S.documents()     # both, plus prose around either
    S.file_texts()    # documents() without surrogates (they cannot be written to a UTF-8 file)

Hypothesis never generates surrogates unless asked, so ``st.characters(categories=["Cs"])`` and ``\\uD8xx``
escapes are drawn explicitly. Look-alike and invisible characters are spelled as escapes on purpose.
"""

from __future__ import annotations

import json
import string
from typing import Any

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

STRUCTURE = ("{", "}", "[", "]", ":", ",", "(", ")", "{}", "[]", "},", "],", '":', '":"', ", ")
QUOTES = (
    '"',
    "'",
    "\N{LEFT DOUBLE QUOTATION MARK}",
    "\N{RIGHT DOUBLE QUOTATION MARK}",
    "\N{DOUBLE LOW-9 QUOTATION MARK}",
    "\N{LEFT SINGLE QUOTATION MARK}",
    "\N{RIGHT SINGLE QUOTATION MARK}",
    "`",
    "\N{LEFT-POINTING DOUBLE ANGLE QUOTATION MARK}",
    "\N{RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK}",
    "\N{FULLWIDTH QUOTATION MARK}",
    '""',
    "''",
    '"""',
)
ESCAPE_LITERALS = (
    '\\"',
    "\\\\",
    "\\/",
    "\\b",
    "\\f",
    "\\n",
    "\\r",
    "\\t",
    "\\'",
    "\\a",
    "\\0",
    "\\",
    "\\x41",
    "\\x",
    "\\u",
    "\\u12",
    "\\u00e9",
    "\\u0000",
    "\\ud83d\\ude00",  # an escaped surrogate pair
    "\\ude00\\ud83d",  # reversed
    "\\ud800",  # lone high
    "\\udc00x",  # lone low
    "\\uDFFF",
    "\\U0001F600",
    "\\\\n",
)
LITERALS = (
    "true",
    "false",
    "null",
    "True",
    "False",
    "None",
    "TRUE",
    "nUll",
    "NaN",
    "Infinity",
    "-Infinity",
    "undefined",
    # Prefixes and overlong forms (a stream cut mid-literal, a stutter), built so spell checkers stay quiet.
    *(lit[:k] for lit in ("true", "false", "null") for k in range(1, len(lit))),
    *(lit + lit[-1] for lit in ("true", "false", "null")),
)
NUMBER_FORMS = (
    "0",
    "-0",
    "1",
    "-1",
    "1.5",
    "-.5",
    ".5",
    "1.",
    "1e",
    "1e5",
    "1E+5",
    "1e-5",
    "-2.5E+3",
    "1_000",
    "1,234",
    "1/2",
    "10-20",
    "1.1.1",
    "0x1F",
    "007",
    "+1",
    "1e+",
    "1e-",
    "12345678901234567890",
    "-9223372036854775809",
    "1e400",
    "-0.0",
    "5e-324",
    "9" * 40,
    "2\N{SUPERSCRIPT TWO}",
    "\N{ARABIC-INDIC DIGIT THREE}",
    "\N{FULLWIDTH DIGIT ONE}\N{FULLWIDTH DIGIT TWO}",
    "-",
    ".",
    "e",
)
COMMENTS = ("# note\n", "// note\n", "/* note */", "/* open", "#", "//", "/*", "*/", "<!-- x -->", "# {\n")
FENCES = ("```json\n", "```\n", "```", "``` json", "```JSON\n", "\n```", "\n```\n", "~~~")
WHITESPACE = (
    " ",
    "  ",
    "\t",
    "\n",
    "\r",
    "\r\n",
    "\x0b",
    "\x0c",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x1f",
    "\x85",
    "\xa0",
    "\N{LINE SEPARATOR}",
    "\N{PARAGRAPH SEPARATOR}",
    "\N{IDEOGRAPHIC SPACE}",
    "\N{ZERO WIDTH NO-BREAK SPACE}",
    "\N{ZERO WIDTH SPACE}",
    "\N{EN QUAD}",
)
LETTERS = (
    "\xe9",
    "\xdf",
    "\N{CJK UNIFIED IDEOGRAPH-4E2D}",
    "\N{HIRAGANA LETTER A}",
    "\N{CYRILLIC CAPITAL LETTER ZHE}",
    "\N{LATIN SMALL LIGATURE FI}",
    "\N{LATIN CAPITAL LETTER I WITH DOT ABOVE}",
    "\N{LATIN CAPITAL LETTER D WITH SMALL LETTER Z WITH CARON}",
    "\N{GREEK CAPITAL LETTER SIGMA}",
    "\N{ANGSTROM SIGN}",
)
WORDS = (
    "key",
    "value",
    "name",
    "a",
    "b",
    "id",
    "items",
    "text",
    "json",
    "Here is the JSON:",
    "Sure!",
    "lorem ipsum",
    "it's",
    "don't",
)

escapes: SearchStrategy[str] = st.one_of(
    st.sampled_from(ESCAPE_LITERALS),
    st.integers(0, 0xFFFF).map(lambda n: f"\\u{n:04x}"),
    st.integers(0xD800, 0xDFFF).map(lambda n: f"\\u{n:04X}"),  # escaped surrogates
)
numbers: SearchStrategy[str] = st.one_of(
    st.sampled_from(NUMBER_FORMS),
    st.integers(-(10**25), 10**25).map(str),
    st.floats().map(repr),  # includes nan, inf, -0.0, 1e+16
)
lone_surrogates: SearchStrategy[str] = st.characters(categories=["Cs"])
identifiers: SearchStrategy[str] = st.one_of(
    st.text(string.ascii_letters + string.digits + "_-$@.", min_size=1, max_size=8),
    st.text(st.characters(categories=["L"]), min_size=1, max_size=6),  # letters of every script
    st.sampled_from(LETTERS),
    st.sampled_from(WORDS),
)
# Decimal digits of every script, superscripts, Roman numerals: str.isdigit() / isdecimal() / isnumeric() differ.
digits: SearchStrategy[str] = st.characters(categories=["N"])

token: SearchStrategy[str] = st.one_of(
    st.sampled_from(STRUCTURE),
    st.sampled_from(QUOTES),
    escapes,
    st.sampled_from(LITERALS),
    numbers,
    st.sampled_from(COMMENTS),
    st.sampled_from(FENCES),
    st.sampled_from(WHITESPACE),
    identifiers,
    digits,
    lone_surrogates,
    st.text(max_size=4),  # anything else (no surrogates: Hypothesis' default)
)


def json_ish(max_tokens: int = 40) -> SearchStrategy[str]:
    """Up to ``max_tokens`` grammar tokens, concatenated."""
    return st.lists(token, max_size=max_tokens).map("".join)


# ------------------------------------------------------------------------------------- near-valid JSON

keys: SearchStrategy[str] = st.one_of(
    st.sampled_from(WORDS), st.text(max_size=6), st.text(st.characters(codec="utf-8"), max_size=4)
)
scalars: SearchStrategy[Any] = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-(10**30), 10**30),
    st.floats(),
    st.text(max_size=12),
    st.lists(st.one_of(lone_surrogates, st.sampled_from(WORDS)), max_size=3).map("".join),
)
values: SearchStrategy[Any] = st.recursive(
    scalars,
    lambda children: st.lists(children, max_size=4) | st.dictionaries(keys, children, max_size=4),
    max_leaves=16,
)


def _dumps(value: Any, indent: int | None, ensure_ascii: bool, python: bool) -> str:
    return repr(value) if python else json.dumps(value, indent=indent, ensure_ascii=ensure_ascii)


serialized: SearchStrategy[str] = st.builds(
    _dumps,
    values,
    st.sampled_from([None, None, 1, 2, 4]),
    st.booleans(),
    st.just(False),
)


def _cut(text: str, at: int) -> str:
    return text[: at % (len(text) + 1)]


def _splice(text: str, piece: str, at: int, drop: int) -> str:
    i = at % (len(text) + 1)
    return text[:i] + piece + text[i + drop :]


def near_json() -> SearchStrategy[str]:
    """Valid JSON (sometimes Python repr) of generated values, then cut, spliced with tokens, or left valid."""
    python_repr = st.builds(_dumps, values, st.none(), st.just(True), st.just(True))
    base = st.one_of(serialized, python_repr)
    return st.one_of(
        base,
        st.builds(_cut, base, st.integers(0, 1 << 16)),
        st.builds(_splice, base, json_ish(3), st.integers(0, 1 << 16), st.integers(0, 3)),
    )


prose: SearchStrategy[str] = st.one_of(
    st.just(""),
    st.sampled_from(("Here is the JSON:\n", "Sure! ", "```json\n", "\n```", " Let me know!", "json\n", "(")),
    st.text(max_size=8),
)


def documents() -> SearchStrategy[str]:
    """``json_ish()``, ``near_json()``, and either wrapped in prose / fences / parentheses."""
    inner = st.one_of(json_ish(), near_json())
    return st.one_of(inner, st.builds(lambda a, b, c: a + b + c, prose, inner, prose))


def _no_surrogates(text: str) -> str:
    return text.encode("utf-8", "replace").decode("utf-8")


def file_texts() -> SearchStrategy[str]:
    """``documents()`` with lone surrogates replaced by '?' (a UTF-8 file cannot hold them)."""
    return documents().map(_no_surrogates)


def options(*names: str) -> SearchStrategy[dict[str, Any]]:
    """Any subset of the boolean keyword arguments ``names`` (often none)."""
    return st.fixed_dictionaries({}, optional={n: st.booleans() for n in names})
