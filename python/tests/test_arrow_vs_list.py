"""Every function that returns Arrow strings gives, row by row, what it gives for a list.

The Arrow path builds UTF-8 buffers straight from the Rust results, so an internal encoding
(json_repair stores lone surrogates in U+10F800..U+10FFFF) could reach the column without
passing through a Python str. Where the list result holds a lone surrogate, which Arrow cannot
store, the Arrow call must raise ``UnicodeEncodeError`` naming the first such row.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

import pyroid.dedup
import pyroid.filters
import pyroid.ftfy
import pyroid.json_repair
import pyroid.nemo
import pyroid.text
import pyroid.translit
from pyroid._pyroid import _testing

# Pieces that steer each module into its interesting paths: JSON syntax and escapes
# (json_repair keeps \ud800 as a lone surrogate), mojibake and CESU-8 (ftfy), Markdown,
# URLs, e-mail addresses, line structure, non-ASCII scripts.
PIECES = [
    "{",
    "}",
    "[",
    "]",
    '"',
    "'",
    ",",
    ":",
    "\\ud800",
    "\\udc00",
    "\\ud83d\\ude00",
    "\\u00e9",
    "\\n",
    "true",
    "1e5",
    "Ã©",
    "â€™",
    "\xed\xa0\xbd\xed\xb8\x80",
    "&amp;",
    "**",
    "_",
    "[a](http://b.c)",
    "http://example.com/x",
    "a@b.com",
    "10.0.0.1",
    "\n",
    "\n\n\n",
    "\r\n",
    "  ",
    "\t",
    "lorem ipsum",
    "javascript",
    "...",
    "é",
    "日本語",
    "😀",
    "Straße",
    "word",
    ".",
]
TEXT = st.lists(
    st.sampled_from(PIECES) | st.text(st.characters(exclude_categories=["Cs"]), max_size=6),
    max_size=12,
).map("".join)
DOCS = st.lists(TEXT, min_size=1, max_size=8)

TRANSFORMS: dict[str, Callable[[Any], Any]] = {
    "text.normalize_whitespace": pyroid.text.normalize_whitespace,
    "ftfy.fix_text_batch": pyroid.ftfy.fix_text_batch,
    "ftfy.fix_text_batch(fix_surrogates=False)": lambda x: pyroid.ftfy.fix_text_batch(
        x, fix_surrogates=False
    ),
    "translit.anyascii": pyroid.translit.anyascii,
    "translit.unidecode": pyroid.translit.unidecode,
    "json_repair.repair_json_batch": pyroid.json_repair.repair_json_batch,
    "json_repair.repair_json_batch(ensure_ascii=False)": lambda x: (
        pyroid.json_repair.repair_json_batch(x, ensure_ascii=False)
    ),
    "json_repair.repair_json_batch(ensure_ascii=False, skip_json_loads=True)": lambda x: (
        pyroid.json_repair.repair_json_batch(x, ensure_ascii=False, skip_json_loads=True)
    ),
    "nemo.remove_markdown": pyroid.nemo.remove_markdown,
    "nemo.normalize_newlines": pyroid.nemo.normalize_newlines,
    "nemo.remove_quotation": pyroid.nemo.remove_quotation,
    "nemo.remove_urls": pyroid.nemo.remove_urls,
    "nemo.remove_boilerplate_strings": pyroid.nemo.remove_boilerplate_strings,
    "filters.c4_lines": pyroid.filters.c4_lines,
    "filters.symbol_lines": pyroid.filters.symbol_lines,
    "filters.pii_replace": pyroid.filters.pii_replace,
    "filters.gopher_quality_reasons": pyroid.filters.gopher_quality_reasons,
    "dedup.simplify_text": pyroid.dedup.simplify_text,
}


def encodable(s: str | None) -> bool:
    try:
        if s is not None:
            s.encode()
    except UnicodeEncodeError:
        return False
    return True


@pytest.mark.parametrize("name", list(TRANSFORMS))
@pytest.mark.parametrize("large", [False, True])
@given(docs=DOCS, null_at=st.integers(0, 8))
def test_arrow_rows_equal_list_items(name: str, large: bool, docs: list[str], null_at: int) -> None:
    f = TRANSFORMS[name]
    with warnings.catch_warnings():
        # json_repair delegates a document holding U+10F800..U+10FFFF (FallbackWarning).
        warnings.simplefilter("ignore")
        want = f(docs)
    rows: list[str | None] = list(docs)
    rows.insert(min(null_at, len(rows)), None)
    expected: list[str | None] = list(want)
    expected.insert(min(null_at, len(rows) - 1), None)
    col = _testing.column(rows, large=large, chunks=2)
    bad = [i for i, s in enumerate(expected) if not encodable(s)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if bad:
            with pytest.raises(UnicodeEncodeError) as info:
                f(col)
            assert any(f"row {bad[0]} of the Arrow input" in n for n in info.value.__notes__)
            return
        got = f(col)
    assert got.to_pylist() == expected
    assert got.type in ("string", "large_string")
