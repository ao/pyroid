"""Generate `tables.rs`, the code point classes behind `pyroid_core::nltk::chars`.

NLTK 3.10.4 compiles almost every tokenizer pattern with the third-party `regex` module (through
`nltk.redos`), and a few (the Treebank contractions) with the standard `re`. Around the patterns
it calls `str.lower()`, `str.isupper()`, `str.islower()` and `str.strip()`. So two sources decide
what a character is:

* the `regex` release (here 2026.9.29, the one pyroid's test group pins), for `\\w` and `\\d` in
  NLTK's `regex` patterns. Its Unicode tables are its own, not the interpreter's;
* CPython itself, one interpreter per Unicode version, for `isupper` and `islower`
  (`re`'s `\\w` and `str.lower()` come from `pyroid_core::filters`).

`dump` records, under one interpreter, every code point's answers, and checks the facts the Rust
side hard-codes: `regex`'s `\\s` is `str.isspace()` minus U+001C..U+001F, `\\W`/`\\D`/`\\S` are
the complements, and the `(?i)` letter sets of `regex` and `re` for the letters NLTK's
case-insensitive patterns use. `emit` writes the tables (one per distinct CPython table).

Usage (in the dev container; `/target/venv-$v` stands for a venv per CPython version with
`regex==2026.9.29` installed):

    for v in 311 312 313 314 315; do
        /target/venv-$v/bin/python generate_tables.py dump /target/gen-nltk/$v.json
    done
    /target/venv-315/bin/python generate_tables.py emit /target/gen-nltk/*.json

The module is declared `#[rustfmt::skip]`: one line per table keeps the file small.
"""

# ruff: noqa: T201  (a build script: it reports what it wrote on stdout)

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import regex

HERE = Path(__file__).resolve().parent
REGEX_VERSION = "2026.9.29"
MAX_CP = 0x110000
VERSIONS = {
    "14.0.0": "V14_0",
    "15.0.0": "V15_0",
    "15.1.0": "V15_1",
    "16.0.0": "V16_0",
    "17.0.0": "V17_0",
}
# The letters of NLTK's case-insensitive patterns (MacIntyre contractions, the clitic lookahead),
# and the code points `(?i)` matches for each, besides the letter and its ASCII other case.
CASE_LETTERS = "acdegilmnorstvwy"
EXTRA_CASES = {
    "regex": {"i": [0x130], "s": [0x17F]},
    "re": {"i": [0x130, 0x131], "s": [0x17F]},
}


def _chars():
    for cp in range(MAX_CP):
        if not 0xD800 <= cp < 0xE000:
            yield cp, chr(cp)


def _ranges(hits) -> list[list[int]]:
    out: list[list[int]] = []
    for cp in sorted(hits):
        if out and out[-1][1] == cp - 1:
            out[-1][1] = cp
        else:
            out.append([cp, cp])
    return out


def _check_case_sets() -> None:
    for name, mod in (("regex", regex), ("re", re)):
        for letter in CASE_LETTERS:
            pat = mod.compile("(?i)" + letter)
            got = sorted(cp for cp, ch in _chars() if pat.fullmatch(ch))
            want = sorted(
                [ord(letter), ord(letter.upper()), *EXTRA_CASES[name].get(letter, [])]
            )
            assert got == want, (name, letter, [hex(c) for c in got])


def dump(path: str) -> None:
    assert regex.__version__ == REGEX_VERSION, regex.__version__
    rx = {k: regex.compile("\\" + k) for k in "wdsWDS"}
    word, digit, upper, lower = set(), set(), set(), set()
    for cp, ch in _chars():
        w, d, s = (rx[k].match(ch) is not None for k in "wds")
        assert (rx["W"].match(ch) is None) == w and (rx["D"].match(ch) is None) == d
        assert (rx["S"].match(ch) is None) == s
        assert s == (ch.isspace() and not 0x1C <= cp <= 0x1F), hex(cp)
        if w:
            word.add(cp)
        if d:
            digit.add(cp)
        if ch.isupper():
            upper.add(cp)
        if ch.islower():
            lower.add(cp)
    assert digit <= word
    _check_case_sets()
    data = {
        "python": sys.version.split()[0],
        "unicode": unicodedata.unidata_version,
        "regex": regex.__version__,
        "rx_word": _ranges(word),
        "rx_digit": _ranges(digit),
        "upper": _ranges(upper),
        "lower": _ranges(lower),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, sort_keys=True))
    print(f"wrote {path}: Python {data['python']}, Unicode {data['unicode']}")


def _ascii_mask(ranges) -> int:
    mask = 0
    for a, b in ranges:
        for cp in range(a, min(b, 0x7F) + 1):
            mask |= 1 << cp
    return mask


def _table(name: str, ranges) -> str:
    rest = [(max(a, 0x80), b) for a, b in ranges if b >= 0x80]
    body = ", ".join(f"(0x{a:x}, 0x{b:x})" for a, b in rest)
    return f"pub(super) static {name}: [(u32, u32); {len(rest)}] = [{body}];\n"


def emit(paths: list[str]) -> None:
    dumps = [json.loads(Path(p).read_text()) for p in paths]
    by_version = {d["unicode"]: d for d in dumps}
    assert sorted(by_version) == sorted(VERSIONS), sorted(by_version)
    first = dumps[0]
    for d in dumps:  # regex's tables do not depend on the interpreter
        assert d["rx_word"] == first["rx_word"] and d["rx_digit"] == first["rx_digit"]
    out = [
        "// @generated by generate_tables.py -- do not edit by hand.\n",
        f"// `regex` {REGEX_VERSION}'s \\w and \\d; CPython's str.isupper() / str.islower() per "
        "Unicode version.\n\n",
        f"pub(super) const RX_WORD_ASCII: u128 = 0x{_ascii_mask(first['rx_word']):x};\n",
        _table("RX_WORD", first["rx_word"]),
        f"pub(super) const RX_DIGIT_ASCII: u128 = 0x{_ascii_mask(first['rx_digit']):x};\n",
        _table("RX_DIGIT", first["rx_digit"]),
    ]
    for prop in ("upper", "lower"):
        masks = {_ascii_mask(d[prop]) for d in dumps}
        assert len(masks) == 1, prop
        out.append(f"pub(super) const {prop.upper()}_ASCII: u128 = 0x{masks.pop():x};\n")
        names: dict[str, str] = {}
        arms = []
        for uni, variant in VERSIONS.items():
            key = json.dumps(by_version[uni][prop])
            if key not in names:
                names[key] = f"{prop.upper()}_{variant}"
                out.append(_table(names[key], by_version[uni][prop]))
            arms.append(f"        UnicodeVersion::{variant} => &{names[key]},\n")
        out.append(
            f"pub(super) fn {prop}(v: UnicodeVersion) -> &'static [(u32, u32)] {{\n"
            "    match v {\n" + "".join(arms) + "    }\n}\n"
        )
    text = "".join(out).replace(
        "\n\n", "\n\nuse crate::ftfy::UnicodeVersion;\n\n", 1
    )
    (HERE / "tables.rs").write_text(text)
    print(f"wrote {HERE / 'tables.rs'}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["dump"] and len(sys.argv) == 3:
        dump(sys.argv[2])
    elif sys.argv[1:2] == ["emit"] and len(sys.argv) > 2:
        emit(sys.argv[2:])
    else:
        sys.exit(__doc__)
