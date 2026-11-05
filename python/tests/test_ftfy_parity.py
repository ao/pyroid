# Derived from ftfy 6.3.1 (Apache-2.0), © Robyn Speer; modified.
"""Hypothesis differential tests: pyroid.ftfy must equal ftfy 6.3.1 exactly.

Hypothesis never generates surrogates unless asked, so the strategies opt in
to them explicitly, along with C1 controls, every byte of every sloppy codec,
HTML entities, terminal escapes, combining marks and characters new in
Unicode 15-17. Set HYPOTHESIS_PROFILE=ci (1000 examples per test) or deep (20000).
"""

# The test data is full of look-alike characters on purpose.
# ruff: noqa: RUF001

from __future__ import annotations

import os
import warnings
from typing import Any

import pytest

from pyroid import ftfy as candidate

ftfy: Any = pytest.importorskip("ftfy")
hypothesis = pytest.importorskip("hypothesis")
if ftfy.__version__ != "6.3.1":
    pytest.skip(f"needs ftfy 6.3.1, found {ftfy.__version__}", allow_module_level=True)

import ftfy.bad_codecs  # noqa: E402  (registers the sloppy codecs)
from ftfy.chardata import CHARMAP_ENCODINGS  # noqa: E402
from hypothesis import HealthCheck, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

settings.register_profile(
    "ci", max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile(
    "deep", max_examples=20000, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile(
    "dev", max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


def outcome(fn: Any, *args: Any, **kwargs: Any) -> tuple[str, Any]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ("ok", fn(*args, **kwargs))
    except Exception as e:
        return ("raise", type(e).__name__)


def garble(s: str, enc: str, layers: int) -> str:
    """UTF-8 bytes misread as a legacy charset, `layers` times."""
    for _ in range(layers):
        s = s.encode("utf-8", "surrogatepass").decode(enc, errors="replace")
    return s


SLOPPY_BYTES = sorted({bytes([b]).decode(e) for e in CHARMAP_ENCODINGS for b in range(0x80, 0x100)})
NEW_CHARS = [
    chr(c)
    for r in (
        (0x11F00, 0x11F5A),
        (0x1E5D0, 0x1E5FB),
        (0x16D40, 0x16D79),
        (0x11380, 0x113E2),
        (0x2EBF0, 0x2EBF8),
        (0x11DB0, 0x11DE9),
    )
    for c in range(*r)
]

words = st.text(st.characters(categories=("L", "M", "N", "P", "S", "Zs")), max_size=30)
astral = st.characters(min_codepoint=0x10000, exclude_categories=["Cs"])
surrogate_pair = astral.map(
    lambda c: chr(0xD800 + ((ord(c) - 0x10000) >> 10)) + chr(0xDC00 + ((ord(c) - 0x10000) & 0x3FF))
)
specials = st.sampled_from(
    [
        "&amp;",
        "&eacute;",
        "&EACUTE;",
        "&#x1F600;",
        "&#59;",
        "&#0;",
        "&#x110000;",
        "&lt;",
        "&amp;amp;",
        "&NTILDE;",
        "&nTILDE;",
        "&SZLIG;",
        "&" + "a" * 25 + ";",
        "\x1b[31m",
        "\x1b[1;2;3m",
        "\x1b[",
        "\x1b[١٢m",
        "“”‘’ʼ",
        "ﬁ",
        "ＡＢＣ",
        "ｶﾞ",
        "\r\n",
        "\r",
        " ",
        "\x85",
        "\x92",
        "﻿",
        "\x00",
        "\x1a",
        "\xa0",
        "Ã ",
        "Ã",
        "<",
        "\n",
        "\t",
        "́",
        "가",
        "ᅡ",
        "Ã©",
        "â€™",
        "Ã la",
        "fÃ cil",
        "вЂ",
        "ГўВЂВ",
        "Ã\n",
        "À\n",
    ]
)
pieces = st.one_of(
    words,
    st.builds(
        garble, words, st.sampled_from([*CHARMAP_ENCODINGS, "windows-1252"]), st.integers(1, 2)
    ),
    specials,
    st.sampled_from(SLOPPY_BYTES),
    st.sampled_from(NEW_CHARS),
    st.characters(categories=["Cs"]),
    surrogate_pair,
    st.characters(min_codepoint=0x80, max_codepoint=0x9F),
    st.characters(categories=("Mn", "Mc", "Me")),
    st.text(st.characters(exclude_categories=["Cs"]), max_size=10),
)
texts = st.lists(pieces, max_size=10).map("".join)

flags = {
    f: st.booleans() for f, d in ftfy.TextFixerConfig._field_defaults.items() if isinstance(d, bool)
}
configs = st.fixed_dictionaries(
    {},
    optional=flags
    | {
        "normalization": st.sampled_from(["NFC", "NFD", "NFKC", "NFKD", None]),
        "unescape_html": st.sampled_from(["auto", True, False]),
        "max_decode_length": st.integers(1, 40),
    },
)
FUNCTIONS = [
    "fix_text",
    "fix_text_segment",
    "fix_and_explain",
    "fix_encoding",
    "fix_encoding_and_explain",
]


@given(text=texts, kwargs=configs, fn=st.sampled_from(FUNCTIONS))
@example(text="voilÃ  le travail", kwargs={}, fn="fix_text")
@example(text="😀 lone \udc00", kwargs={}, fn="fix_text")
@example(text="💩\udca9\ud83d", kwargs={"fix_surrogates": False}, fn="fix_text")
@example(text="Ã©À\n", kwargs={}, fn="fix_text")
@example(text="x\xed\xa0\xbd\xed\xb8\n", kwargs={}, fn="fix_encoding")
def test_matches_ftfy(text: str, kwargs: dict[str, Any], fn: str) -> None:
    assert outcome(getattr(candidate, fn), text, **kwargs) == outcome(
        getattr(ftfy, fn), text, **kwargs
    )


@given(text=texts, explain=st.booleans())
def test_config_object_matches_ftfy(text: str, explain: bool) -> None:
    ours, theirs = candidate.TextFixerConfig(explain=explain), ftfy.TextFixerConfig(explain=explain)
    assert outcome(candidate.fix_text, text, ours) == outcome(ftfy.fix_text, text, theirs)
    assert outcome(candidate.fix_and_explain, text, ours) == outcome(
        ftfy.fix_and_explain, text, theirs
    )


@given(text=st.text(st.characters(codec="utf-8"), max_size=50))
def test_any_valid_text_matches_ftfy(text: str) -> None:
    assert candidate.fix_text(text) == ftfy.fix_text(text)
