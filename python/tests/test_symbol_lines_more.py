"""pyroid.filters.symbol_lines: datatrove 0.10.1's SymbolLinesFormatter, byte for byte."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pyroid
import pyroid.filters as pf
from pyroid._pyroid import _testing


def reference(text: str, symbols: Iterable[str], replace_char: str = "") -> str:
    """datatrove's ``SymbolLinesFormatter.format``, transcribed."""
    symbols = set(symbols)
    formatted = []
    in_removed_span = False
    for line in text.splitlines():
        chars_line = line.strip() != "" and all(c in symbols or c == " " for c in line)
        if chars_line and not in_removed_span:
            if replace_char:
                formatted.append(replace_char)
            in_removed_span = True
        elif not chars_line:
            formatted.append(line)
            in_removed_span = False
    return "\n".join(formatted)


PIECES = [
    "|",
    "-",
    "*",
    "a",
    "1",
    " ",
    "\t",
    "\n",
    "\r\n",
    "\r",
    "\x0b",
    "\x85",
    "\u2028",
    "—",
    "»",
    "…",
    "é",
    "  ",
]
TEXT = st.lists(st.sampled_from(PIECES), max_size=30).map("".join)
SYMBOLS = st.lists(st.sampled_from(["|", "-", "*", "—", "ab", "", "\t"]), max_size=4)


@settings(max_examples=400, deadline=None, suppress_health_check=list(HealthCheck))
@given(st.lists(TEXT, min_size=1, max_size=5), SYMBOLS, st.sampled_from(["", "\n", "<sym>"]))
def test_matches_reference(texts: list[str], symbols: list[str], replace_char: str) -> None:
    got = pf.symbol_lines(texts, symbols_to_remove=symbols, replace_char=replace_char)
    assert got == [reference(t, symbols, replace_char) for t in texts]


@settings(max_examples=300, deadline=None, suppress_health_check=list(HealthCheck))
@given(st.lists(st.one_of(TEXT, st.text(max_size=20)), max_size=5))
def test_default_symbols_match_live_datatrove(texts: list[str]) -> None:
    sym = pytest.importorskip("datatrove.pipeline.formatters.symbol_lines_remover")
    texts = [t.encode("utf-8", "replace").decode() for t in texts]
    cases: list[dict[str, Any]] = [{}, {"symbols_to_remove": ["|"], "replace_char": "\n"}]
    for kw in cases:
        ref = sym.SymbolLinesFormatter(**kw)
        assert pf.symbol_lines(texts, **kw) == [ref.format(t) for t in texts]


def test_examples() -> None:
    assert pf.symbol_lines("a\n---\n***\nb") == "a\nb"
    assert pf.symbol_lines(
        "| x |\n|---|\n| y |", symbols_to_remove=["|", "-"], replace_char="\n"
    ) == ("| x |\n\n\n| y |")
    assert pf.symbol_lines("a\r\nb\n") == "a\nb"


@pytest.mark.parametrize("kind", [list, tuple])
def test_container_kinds_and_identity(kind: Callable[..., Any]) -> None:
    texts = kind(["a\nb", "a\n---\nb", ""])
    out = pf.symbol_lines(texts)
    assert type(out) is kind
    assert out[0] is texts[0]
    assert out[2] is texts[2]
    assert out[1] == "a\nb"


@pytest.mark.parametrize("large", [False, True])
def test_arrow(large: bool) -> None:
    values = ["a\n---\nb", None, "ok"]
    col = pf.symbol_lines(_testing.column(values, large=large, chunks=2, name="t"))
    assert col.to_pylist() == ["a\nb", None, "ok"]
    assert col.type == ("large_string" if large else "string")


def test_errors() -> None:
    with pytest.raises(pyroid.InputError):
        pf.symbol_lines(b"x")  # type: ignore[call-overload]
    with pytest.raises(TypeError):
        pf.symbol_lines("x", replace_char=1)  # type: ignore[call-overload]
    with pytest.raises(UnicodeEncodeError):
        pf.symbol_lines(["\ud800"])


@pytest.mark.parametrize("threads", [None, 1, 2])
def test_threads(threads: int | None) -> None:
    docs = [f"line {i}\n{'-' * (i % 5)}\nend" for i in range(300)]
    assert pf.symbol_lines(docs, threads=threads) == [reference(d, "-*|") for d in docs]


def test_drop_in_step_matches_datatrove() -> None:
    sym = pytest.importorskip("datatrove.pipeline.formatters.symbol_lines_remover")
    data = pytest.importorskip("datatrove.data")
    from pyroid.integrations import datatrove as pdt

    texts = ["a\n|\n| |\nb", "x\n---\ny", "", "\n\n|\n"] * 5
    for kw in ({}, {"symbols_to_remove": ["|"], "replace_char": "\n"}):
        stock = sym.SymbolLinesFormatter(**kw)
        mine = pdt.SymbolLinesFormatter(**kw, batch_size=3)
        docs = [data.Document(text=t, id=str(i)) for i, t in enumerate(texts)]
        want = [d.text for d in stock.run([data.Document(text=t, id="") for t in texts])]
        assert [d.text for d in mine.run(docs)] == want
        assert mine.format(texts[0]) == stock.format(texts[0])


def test_module_surface() -> None:
    assert {"SymbolLines", "symbol_lines"} <= set(pf.__all__)
    assert pf.symbol_lines.__module__ == "pyroid.filters"
