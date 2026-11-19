"""pyroid.tabulate: formats, settings, fallbacks, patching, threads.

The exhaustive comparison with tabulate is ``test_tabulate_parity.py`` (Hypothesis) and
``test_tabulate_upstream.py`` (tabulate's own tests); these tests pin the behaviour that does
not need tabulate installed, plus the pyroid-specific parts.
"""

from __future__ import annotations

import ast
import concurrent.futures
import importlib.util
import inspect
import os
import signal
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import pytest

import pyroid
import pyroid.tabulate
from pyroid.tabulate import tabulate

# The private helpers are not in the stub.
pt: Any = pyroid.tabulate


def _real_tabulate() -> Any:
    mod = pytest.importorskip("tabulate")
    if getattr(mod, "__pyroid__", False) or mod.__version__ != pt.TABULATE_VERSION:
        pytest.skip(f"needs the real tabulate {pt.TABULATE_VERSION}")
    return mod


def test_readme_example() -> None:
    table = [["Sun", 696000, 1989100000], ["Earth", 6371, 5973.6], ["Moon", 1737, 73.5]]
    assert tabulate(table, headers=["Planet", "R (km)", "mass (x 10^29 kg)"]) == (
        "Planet      R (km)    mass (x 10^29 kg)\n"
        "--------  --------  -------------------\n"
        "Sun         696000           1.9891e+09\n"
        "Earth         6371        5973.6\n"
        "Moon          1737          73.5"
    )


def test_signature_is_tabulates() -> None:
    params = inspect.signature(tabulate).parameters
    assert list(params) == [
        "tabular_data",
        "headers",
        "tablefmt",
        "floatfmt",
        "intfmt",
        "numalign",
        "stralign",
        "missingval",
        "showindex",
        "disable_numparse",
        "colglobalalign",
        "colalign",
        "preserve_whitespace",
        "maxcolwidths",
        "headersglobalalign",
        "headersalign",
        "rowalign",
        "maxheadercolwidths",
        "break_long_words",
        "break_on_hyphens",
    ]
    assert inspect.isfunction(tabulate)
    real = importlib.util.find_spec("tabulate")
    if real is not None:
        mod = _real_tabulate()
        want = inspect.signature(mod.tabulate).parameters
        assert [(p.name, p.default) for p in params.values()] == [
            (p.name, p.default) for p in want.values()
        ]


def test_star_import_is_tabulates() -> None:
    # TABULATE_VERSION and the other extras stay attributes, out of `from ... import *`.
    assert pt.__all__ == ["tabulate", "tabulate_formats", "simple_separated_format"]
    if importlib.util.find_spec("tabulate") is not None:
        assert _real_tabulate().__all__ == pt.__all__


def test_formats_list() -> None:
    assert pt.tabulate_formats == sorted(pt.tabulate_formats)
    assert len(pt.tabulate_formats) == 37
    assert "colon_grid" in pt.tabulate_formats
    assert pt.tabulate_formats == pyroid._pyroid.tabulate.FORMAT_NAMES


@pytest.mark.parametrize("fmt", pt.tabulate_formats)
def test_every_format_renders(fmt: str) -> None:
    out = tabulate([["spam", 41.9999], ["eggs", "451.0"]], ["strings", "numbers"], fmt)
    assert "41.9999" in out
    assert "451" in out
    html = fmt in ("html", "unsafehtml")
    assert type(out) is (pt.JupyterHTMLStr if html else str)


def test_known_outputs() -> None:
    data = [["spam", 41.9999], ["eggs", "451.0"]]
    assert tabulate(data, ["strings", "numbers"], "pipe") == (
        "| strings   |   numbers |\n"
        "|:----------|----------:|\n"
        "| spam      |   41.9999 |\n"
        "| eggs      |  451      |"
    )
    assert tabulate(data, tablefmt="latex") == (
        "\\begin{tabular}{lr}\n\\hline\n spam &  41.9999 \\\\\n eggs & 451      \\\\\n"
        "\\hline\n\\end{tabular}"
    )
    assert tabulate([["a&b", "<i>"]], tablefmt="html") == (
        "<table>\n<tbody>\n<tr><td>a&amp;b</td><td>&lt;i&gt;</td></tr>\n</tbody>\n</table>"
    )
    assert tabulate([[1, 2.5]], headers=["x", "y"], tablefmt="colon_grid") == (
        "+-----+-----+\n| x   | y   |\n+:====+:====+\n| 1   | 2.5 |\n+-----+-----+"
    )
    assert tabulate([], headers=[]) == ""


def test_number_parsing_like_tabulate() -> None:
    # 0.10.0: blanks count as missing, thousands separators parse, git SHAs look like floats.
    assert tabulate([["1"], [""], ["10"]]) == "--\n 1\n\n10\n--"
    assert tabulate([["1,000"], ["2"]]) == "-----\n1,000\n    2\n-----"
    assert tabulate([["42992e1"]]) == "------\n429920\n------"
    assert tabulate([["42992e1"]], disable_numparse=True) == "-------\n42992e1\n-------"


def test_separating_line() -> None:
    out = tabulate([[1, "a"], pt.SEPARATING_LINE, [2, "b"]])
    assert out == "-  -\n1  a\n-  -\n2  b\n-  -"


def test_multiline_and_wide() -> None:
    out = tabulate([["a\nbb", "日本"]], tablefmt="grid")
    assert out == "+----+------+\n| a  | 日本 |\n| bb |      |\n+----+------+"


def test_ansi_codes_are_invisible() -> None:
    red = "\x1b[31mred\x1b[0m"
    assert tabulate([[red, 1]], tablefmt="plain") == f"{red}  1"


@pytest.mark.thread_unsafe  # a module setting
def test_min_padding_setting() -> None:
    saved = pt.MIN_PADDING
    try:
        pt.MIN_PADDING = 0
        assert tabulate([[1]], headers=["h"]) == "h\n-\n1"
    finally:
        pt.MIN_PADDING = saved
    assert tabulate([[1]], headers=["h"]) == "  h\n---\n  1"


def test_numpy_and_pandas() -> None:
    np = pytest.importorskip("numpy")
    out = tabulate(np.array([[1, 2], [3, 4]]), headers="keys")
    assert out == "  0    1\n---  ---\n  1    2\n  3    4"
    # tabulate types numpy's unsigned ints as floats, like any number float() accepts.
    assert tabulate([[np.uint64(12345678)]]) == "-----------\n1.23457e+07\n-----------"
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1, 2], "b": [0.5, None]}, index=["x", "y"])
    assert tabulate(df, headers="keys", tablefmt="pipe") == (
        "|    |   a |     b |\n|:---|----:|------:|\n| x  |   1 |   0.5 |\n| y  |   2 | nan   |"
    )


def test_slow_path_cells() -> None:
    import datetime
    import decimal
    import fractions

    out = tabulate(
        [[decimal.Decimal("1.10"), fractions.Fraction(1, 3), datetime.date(2020, 1, 2), b"xy"]]
    )
    assert out == (
        "---  --------  ----------  --\n"
        "1.1  0.333333  2020-01-02  xy\n"
        "---  --------  ----------  --"
    )


@pytest.mark.thread_unsafe  # warnings
def test_colalign_string_warns_like_tabulate() -> None:
    with pytest.warns(UserWarning, match="As a string, `colalign` is interpreted as"):
        tabulate([[1, 2]], colalign="rl")


def _old_tabulate(path: Path) -> Path:
    """A directory with a stand-in for tabulate 0.9.0 whose ``tabulate()`` must never run."""
    pkg = path / "tabulate"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "__version__ = '0.9.0'\n"
        "MIN_PADDING = 2\n"
        "def tabulate(tabular_data, headers=(), tablefmt='simple', **kwargs):\n"
        "    raise AssertionError('the installed tabulate was called')\n",
        encoding="utf-8",
    )
    return path


# The fallback cases: (args, kwargs, tabulate 0.10.0's text).
_SLOW_PATH_CASES: list[tuple[tuple[Any, ...], dict[str, Any], str | None]] = [
    (([["\ud800"]],), {}, "-\n\ud800\n-"),
    (([[1]],), {"colalign": [3]}, "-\n1\n-"),
    (([["a", 1]],), {"numalign": 5}, "-  -\na  1\n-  -"),
    (((r for r in [["\ud800", 1], ["x", 2]]),), {"headers": ["a", "b"]}, None),
]


@pytest.mark.thread_unsafe  # warnings, sys.modules
@pytest.mark.parametrize("installed", ["none", "0.9.0"])
def test_fallbacks_run_pyroids_copy_of_tabulate(
    installed: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The slow path never needs, or calls, the installed tabulate: it may be missing, or
    # another release whose tabulate() takes other arguments.
    if installed == "none":
        monkeypatch.setitem(sys.modules, "tabulate", None)
    else:
        monkeypatch.syspath_prepend(str(_old_tabulate(tmp_path)))
        monkeypatch.delitem(sys.modules, "tabulate", raising=False)
        old = importlib.import_module("tabulate")
        assert old.__version__ == "0.9.0"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        for args, kwargs, want in _SLOW_PATH_CASES:
            got = tabulate(*args, **kwargs)
            if want is not None:
                assert got == want
        assert tabulate([["\ud800", 1], ["x", 2]], headers=["a", "b"]) == (
            "a      b\n---  ---\n\ud800      1\nx      2"
        )


@pytest.mark.thread_unsafe  # warnings
def test_fallbacks_warn_and_match_tabulate() -> None:
    pt._WARNED.clear()
    with pytest.warns(pyroid.FallbackWarning, match="lone surrogates") as record:
        got = tabulate((r for r in [["\ud800", 1], ["x", 2]]), headers=["a", "b"])
    assert "running tabulate 0.10.0's own Python code" in str(record[0].message)
    assert record[0].filename == __file__
    assert got == "a      b\n---  ---\n\ud800      1\nx      2"
    with warnings.catch_warnings():
        warnings.simplefilter("error", pyroid.FallbackWarning)
        # Shown once per reason.
        tabulate([["\udfff"]])
    custom = pt.TableFormat(None, None, None, None, pt.DataRow("", 5, ""), None, 0, None)
    with (
        pytest.warns(pyroid.FallbackWarning, match="not strings or a callable"),
        pytest.raises(TypeError),
    ):
        tabulate([[1]], tablefmt=custom)
    if importlib.util.find_spec("tabulate") is not None:
        real = _real_tabulate()
        assert got == real.tabulate([["\ud800", 1], ["x", 2]], headers=["a", "b"])


@pytest.mark.thread_unsafe  # module settings, warnings
def test_settings_reach_the_slow_path(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [["\ud800", 1], ["ok", 22]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        monkeypatch.setattr(pt, "MIN_PADDING", 0)
        assert pt._port.MIN_PADDING == 0
        assert tabulate(rows, headers=["h0", "h1"]) == "h0  h1\n--  --\n\ud800    1\nok  22"
        monkeypatch.undo()
        assert pt._port.MIN_PADDING == 2
        assert tabulate(rows, headers=["h0", "h1"]) == (
            "h0      h1\n----  ----\n\ud800        1\nok      22"
        )
        # A non-str alignment falls back too; widths follow WIDE_CHARS_MODE there as well.
        monkeypatch.setattr(pt, "WIDE_CHARS_MODE", False)
        assert tabulate([["日本", 1]], stralign=1.5) == "--  -\n日本  1\n--  -"  # type: ignore[arg-type]
        monkeypatch.undo()
        if pt.wcwidth is not None:
            assert tabulate([["日本", 1]], stralign=1.5) == "----  -\n日本  1\n----  -"  # type: ignore[arg-type]
        monkeypatch.setattr(pt, "wcwidth", None)
        assert pt._port.wcwidth is None
        assert tabulate([["日本", 1]], stralign=1.5) == "--  -\n日本  1\n--  -"  # type: ignore[arg-type]
        monkeypatch.undo()
        # Formats added to `_table_formats` are found by name on both paths.
        brackets = pt.TableFormat(None, None, None, None, None, pt.DataRow("[", "|", "]"), 0, None)
        monkeypatch.setitem(pt._table_formats, "brackets", brackets)
        assert tabulate([["a", 1]], tablefmt="brackets") == "[a|1]"
        assert tabulate([["\ud800", 1]], tablefmt="brackets") == "[\ud800|1]"


@pytest.mark.thread_unsafe  # warnings
def test_lone_surrogates_in_a_custom_format() -> None:
    row = pt.TableFormat(None, None, None, None, None, pt.DataRow("\ud800", "|", ""), 0, None)
    line = pt.TableFormat(
        pt.Line("\udc80", "-", "+", ""),
        None,
        None,
        None,
        pt.DataRow("", "|", ""),
        pt.DataRow("", "|", ""),
        1,
        None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        assert tabulate([["a", 1]], tablefmt=row) == "\ud800a|1"
        assert tabulate([["a", 1]], headers=["x", "y"], tablefmt=line) == (
            "\udc80-----+-----\n x   |   y\n a   |   1"
        )


@pytest.mark.thread_unsafe  # warnings
def test_warnings_point_at_the_caller_on_the_slow_path() -> None:
    pt._WARNED.clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tabulate([["a", 1]], colalign="right", stralign=1.5)  # type: ignore[arg-type]
        tabulate([["\ud800", 1]], headers=["a", "b"], headersalign="r")
        tabulate([["a", 1]], colalign="right")  # the Rust path
    assert [(x.category, x.filename) for x in w] == [
        (pyroid.FallbackWarning, __file__),
        (UserWarning, __file__),
        (pyroid.FallbackWarning, __file__),
        (UserWarning, __file__),
        (UserWarning, __file__),
    ]


def test_float32_frames() -> None:
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    values = [[1.1, 2.2], [3.3, 4.4]]
    f32 = pd.DataFrame(np.array(values, dtype=np.float32))
    f16 = pd.DataFrame(np.array(values, dtype=np.float16))
    # The header row is str() of numpy's scalars, which print their own precision.
    assert tabulate(f32, headers="firstrow") == (
        "  0    1.1    2.2\n---  -----  -----\n  1    3.3    4.4"
    )
    assert tabulate(f16, headers="firstrow") == (
        "  0      1.1      2.2\n---  -------  -------\n  1  3.30078  4.39844"
    )
    # Cells that are not parsed as numbers are f"{v}", which formats the value as a Python float.
    assert tabulate(f32, disable_numparse=True) == (
        "-  -----------------  -----------------\n"
        "0  1.100000023841858  2.200000047683716\n"
        "1  3.299999952316284  4.400000095367432\n"
        "-  -----------------  -----------------"
    )
    assert tabulate(f32, headers="keys", floatfmt=".3f") == (
        "        0      1\n--  -----  -----\n 0  1.100  2.200\n 1  3.300  4.400"
    )


_SEP = pt.SEPARATING_LINE


@pytest.mark.parametrize(
    ("rows", "kwargs", "want"),
    [
        ([["a", _SEP], ["b", 2]], {"showindex": "always"}, "  A\n---\n---\n  0"),
        ([("a", _SEP), ("b", 2)], {"showindex": "always"}, "  A\n---\n---\n  0"),
        ([["a", _SEP], ["b", 2]], {"showindex": [7, 8]}, "  A\n---\n---\n  7"),
        ([["a", _SEP], ["b", 2]], {}, "      A\n--  ---\n--  ---\nb     2"),
        (
            [[_SEP], ["b", 2, 3]],
            {"showindex": True, "headers": ["A", "B"]},
            "  A  B\n---  ---\n---  ---\n  0  b",
        ),
    ],
)
def test_separator_row_first_with_an_index(rows: Any, kwargs: Any, want: str) -> None:
    # tabulate pads the headers to its first row after adding the index, and that row is then
    # SEPARATING_LINE itself.
    kwargs = {"headers": ["A"], **kwargs}
    assert tabulate(rows, **kwargs) == want


def test_separator_row_first_in_a_frame() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"x": ["a", "b"], "y": [_SEP, "2"]}, dtype=object)
    assert tabulate(df, headers=["A"]) == "  A\n---\n---\n  0"


@pytest.mark.parametrize("showindex", ["default", True])
@pytest.mark.parametrize("tablefmt", ["simple", "grid"])
def test_separator_rows_anywhere(showindex: Any, tablefmt: str) -> None:
    # First, last, several in a row, as list, tuple and str rows: each goes back in its place.
    few = [[_SEP], [1, "a"], _SEP, (_SEP, "x"), [2, "b"], [3, "c"], [_SEP], [_SEP]]
    many = [[_SEP] if i % 3 else [i, "a"] for i in range(3_000)]
    for rows in (few, many):
        kwargs = {"headers": ["n", "s"], "tablefmt": tablefmt, "showindex": showindex}
        assert tabulate(rows, **kwargs) == pt._port.tabulate(rows, **kwargs)


@pytest.mark.thread_unsafe  # a module-level table
def test_edited_latex_escape_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    # tabulate's _latex_row reads LATEX_ESCAPE_RULES on each call, so edits to it count.
    rows = [["a&b", "x_y", 1]]
    assert tabulate(rows, tablefmt="latex").splitlines()[2] == " a\\&b & x\\_y & 1 \\\\"
    monkeypatch.setitem(pt.LATEX_ESCAPE_RULES, "&", "AND")
    monkeypatch.delitem(pt.LATEX_ESCAPE_RULES, "_")
    for fmt in ("latex", "latex_booktabs", "latex_longtable"):
        assert tabulate(rows, tablefmt=fmt).splitlines()[2] == " aANDb & x_y & 1 \\\\", fmt
    assert tabulate(rows, tablefmt="latex_raw").splitlines()[2] == " a&b & x_y & 1 \\\\"
    monkeypatch.undo()
    assert tabulate(rows, tablefmt="latex").splitlines()[2] == " a\\&b & x\\_y & 1 \\\\"


def test_custom_table_formats() -> None:
    fmt = pt.TableFormat(
        lineabove=lambda widths, aligns: "=" * sum(w + 1 for w in widths),
        linebelowheader=pt.Line("", "~", " ", ""),
        linebetweenrows=None,
        linebelow=None,
        headerrow=pt.DataRow("<", "|", ">"),
        datarow=lambda cells, widths, aligns: "/".join(c.strip() for c in cells),
        padding=1,
        with_header_hide=None,
    )
    assert tabulate([[1, "x"], [22, "yy"]], headers=["n", "s"], tablefmt=fmt) == (
        "============\n<   n | s   >\n~~~~~ ~~~~~\n1/x\n22/yy"
    )
    assert tabulate([[1, 2]], tablefmt=pt.simple_separated_format(";")) == "1;2"


def _outcome(fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return type(e), str(e)


#: In-place edits to the built-in formats: (format, edit, undo). tabulate reads the
#: partials' arguments and the ``with_header_hide`` lists on every call (P67, `t_mutate_latex`).
def _format_edits(formats: Any) -> list[tuple[str, Any, Any]]:
    f = formats

    def setitem(d: dict[str, Any], key: str, value: Any) -> tuple[Any, Any]:
        missing = object()
        old = d.get(key, missing)

        def undo() -> None:
            if old is missing:
                del d[key]
            else:
                d[key] = old

        return lambda: d.__setitem__(key, value), undo

    def relist(lst: list[str], new: list[str]) -> tuple[Any, Any]:
        old = list(lst)
        return lambda: lst.__setitem__(slice(None), new), lambda: lst.__setitem__(slice(None), old)

    edits = [
        ("latex_raw", *setitem(f["latex_raw"].datarow.keywords["escrules"], "&", "X")),
        ("latex_raw", *setitem(f["latex_raw"].headerrow.keywords["escrules"], "_", "Y")),
        ("latex_booktabs", *setitem(f["latex_booktabs"].lineabove.keywords, "booktabs", False)),
        ("latex_longtable", *setitem(f["latex_longtable"].lineabove.keywords, "booktabs", True)),
        ("moinmoin", *setitem(f["moinmoin"].headerrow.keywords, "header", "*")),
        ("moinmoin", *setitem(f["moinmoin"].datarow.keywords, "header", "~")),
        ("html", *setitem(f["html"].datarow.keywords, "unsafe", False)),  # a TypeError
        ("simple", *relist(f["simple"].with_header_hide, [])),
        ("github", *relist(f["github"].with_header_hide, ["lineabove", "linebelowheader"])),
        ("html", *relist(f["html"].with_header_hide, [])),
        ("asciidoc", *relist(f["asciidoc"].with_header_hide, ["linebelow"])),
    ]
    return edits


def test_builtin_formats_edited_in_place() -> None:
    rows = [["a&b", "x_y", 1.5], ["c", "d\ne", 22]]
    headers = ["h&1", "h_2", "h3"]
    real = _real_tabulate() if importlib.util.find_spec("tabulate") else None
    edits = _format_edits(pt._table_formats)
    real_edits = _format_edits(real._table_formats) if real is not None else [None] * len(edits)
    for (name, do, undo), real_edit in zip(edits, real_edits, strict=True):
        cases = [((rows,), {"tablefmt": name}), ((rows, headers), {"tablefmt": name})]
        before = [tabulate(*a, **k) for a, k in cases]
        do()
        if real_edit is not None:
            real_edit[1]()
        try:
            for a, k in cases:
                got = _outcome(tabulate, *a, **k)
                # tabulate's own code, over the same (edited) `_table_formats`.
                assert got == _outcome(pt._port.tabulate, *a, **k), (name, a, k)
                if real_edit is not None:
                    assert got == _outcome(real.tabulate, *a, **k), (name, a, k)
            assert [_outcome(tabulate, *a, **k) for a, k in cases] != before, name
        finally:
            undo()
            if real_edit is not None:
                real_edit[2]()
        assert [tabulate(*a, **k) for a, k in cases] == before, name
    # The finding's call.
    escrules = pt._table_formats["latex_raw"].datarow.keywords["escrules"]
    escrules["&"] = "X"
    try:
        assert tabulate([["a&b"]], tablefmt="latex_raw").splitlines()[2] == " aXb \\\\"
    finally:
        del escrules["&"]
    assert tabulate([["a&b"]], tablefmt="latex_raw").splitlines()[2] == " a&b \\\\"


class _Recorded:
    """A cell whose ``__str__`` records the order it runs in (and raises for ``bad``)."""

    def __init__(self, calls: list[Any], tag: Any, bad: bool = False) -> None:
        self.calls, self.tag, self.bad = calls, tag, bad

    def __str__(self) -> str:
        self.calls.append(self.tag)
        if self.bad:
            raise ValueError(f"bad {self.tag}")
        return f"c{self.tag}"


def _str_calls(fn: Any, shape: list[list[Any]], **kwargs: Any) -> tuple[Any, list[Any]]:
    """`fn` on a table of `_Recorded` cells shaped like `shape` (a tag per cell, None: None,
    a tag ending in "!": raises), and the order their ``__str__`` ran in."""
    calls: list[Any] = []

    def cell(tag: Any) -> Any:
        if tag is None or isinstance(tag, (int, float)):
            return tag
        return _Recorded(calls, tag, tag.endswith("!"))

    rows = [[cell(t) for t in row] for row in shape]
    return _outcome(fn, rows, **kwargs), calls


def test_cells_str_runs_in_tabulates_order(tablefmt: str) -> None:
    # tabulate takes str() of every cell row by row (to look for ANSI codes), then formats
    # them column by column (P67, `t_str_calls`, `t_bad_str_order`).
    shapes: list[tuple[list[list[Any]], dict[str, Any]]] = [
        ([["a", "b"], ["c", "d"]], {}),
        ([["a", "b", "c"], ["d"], ["e", None, "f"]], {"headers": ["x", "y"]}),
        ([[1, "a"], ["b", 2.5], ["c", "d"]], {"showindex": True}),
        ([[1, "b!"], ["a!", 2]], {}),
        ([["a", "b"], ["c!", "d!"]], {"headers": "firstrow"}),
    ]
    for shape, kwargs in shapes:
        kwargs = {"tablefmt": tablefmt, **kwargs}
        got = _str_calls(tabulate, shape, **kwargs)
        assert got == _str_calls(pt._port.tabulate, shape, **kwargs), (shape, kwargs)
    calls = _str_calls(tabulate, [["1", "2"], ["3", "4"]])[1]
    assert calls == ["1", "2", "3", "4", "1", "3", "2", "4"]
    assert _str_calls(tabulate, [[1, "b!"], ["a!", 2]])[0] == (ValueError, "bad b!")


def test_row_iterators_survive_a_fallback() -> None:
    # The Rust side turns rows into lists as it reads them; when it then hands the call to
    # tabulate's code (here for the lone surrogate in the last row), that code gets those lists,
    # not the spent iterators.
    def rows() -> list[Any]:
        return [[1, 2], iter([3, 4]), (x for x in "ab"), ["\ud800", 5]]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        for kwargs in ({}, {"headers": ["x", "y"]}, {"showindex": True}, {"tablefmt": "rst"}):
            want = pt._port.tabulate(rows(), **kwargs)
            assert "3" in want
            assert tabulate(rows(), **kwargs) == want, kwargs


def test_port_is_tabulates_code() -> None:
    """Every function and table kept in ``_port.py`` is tabulate's, AST for AST."""
    real = _real_tabulate()
    ref_src = inspect.getsource(real)
    port_src = Path(pt._port.__file__).read_text(encoding="utf-8")

    class Strip(ast.NodeTransformer):
        def _clean(self, node: Any) -> Any:
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                node.body = body[1:] or [ast.Pass()]
            self.generic_visit(node)
            return node

        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            node.returns = None
            args = node.args
            for a in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
                if a is not None:
                    a.annotation = None
            return self._clean(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> Any:
            return self._clean(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
            if node.value is None:
                return None
            return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)

    def defs(src: str) -> dict[str, str]:
        out = {}
        for n in ast.parse(src).body:
            name = None
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                name = n.name
            elif isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
                name, n = n.targets[0].id, n.value  # type: ignore[assignment]
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                name, n = n.target.id, n.value  # type: ignore[assignment]
            if name is not None and n is not None:
                out[name] = ast.dump(Strip().visit(n))
        return out

    ref, port = defs(ref_src), defs(port_src)
    differ = [name for name, d in port.items() if name in ref and d != ref[name]]
    assert differ == []
    # All of tabulate but its command line, which pyroid.tabulate runs with its own tabulate().
    assert sorted(set(ref) - set(port)) == ["__all__", "_main", "_pprint_file"]


def test_private_helpers_are_tabulates() -> None:
    # All of tabulate's functions, classes and tables, as the patch shim has them.
    assert [n for n in pt._PATCHED_NAMES if not hasattr(pt, n)] == []
    assert pt._padleft is pt._port._padleft
    assert pt._align_column is pt._port._align_column
    assert pt._align_cell_veritically is pt._port._align_cell_veritically
    assert pt._align_column(["12.3", "1.0", "100"], "decimal") == [" 12.3", "  1.0", "100  "]
    assert pt._align_column(["a", "bb"], "right", 4) == ["   a", "  bb"]
    assert pt._align_cell_veritically(["x"], 3, 2, "bottom") == ["  ", "  ", "x"]
    # With has_invisible, the widths follow WIDE_CHARS_MODE, whatever enable_widechars says.
    wide = pt.wcwidth is not None and pt.WIDE_CHARS_MODE
    want = ["   a", "   b", "日本"] if wide else [" a", " b", "日本"]
    assert pt._align_column(["a", " b ", "日本"], "right", 0, True, False, False, False) == want


@pytest.mark.thread_unsafe  # sys.modules, module settings
def test_patch() -> None:
    from pyroid import integrations

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", integrations.PatchWarning)
        integrations.patch("tabulate")
    try:
        patched: Any = importlib.import_module("tabulate")
        assert patched.__pyroid__
        assert patched.tabulate is pt.tabulate
        assert patched.__version__ == pt.TABULATE_VERSION
        assert patched.TableFormat is pt.TableFormat
        saved = pt.MIN_PADDING
        patched.MIN_PADDING = 0
        try:
            assert pt.MIN_PADDING == 0
            assert patched.tabulate([[1]], headers=["h"]) == "h\n-\n1"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", pyroid.FallbackWarning)
                assert patched.tabulate([["\ud800"]], headers=["h"]) == "h\n-\n\ud800"
        finally:
            patched.MIN_PADDING = saved
        assert saved == pt.MIN_PADDING
        pd = pytest.importorskip("pandas")
        pytest.importorskip("tabulate")
        df = pd.DataFrame({"a": [1, 22]})
        assert df.to_markdown() == pt.tabulate(df, headers="keys", tablefmt="pipe")
    finally:
        integrations.unpatch("tabulate")
    after = importlib.import_module("tabulate") if importlib.util.find_spec("tabulate") else None
    assert not getattr(after, "__pyroid__", False)


@pytest.mark.thread_unsafe  # threads of its own
def test_threads_give_the_same_text() -> None:
    rows = [[i, i * 0.5, f"row {i}", "日本" if i % 3 else None] for i in range(300)]
    want = tabulate(rows, headers=["i", "half", "name", "jp"], tablefmt="grid")
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        outs = list(
            ex.map(
                lambda _: tabulate(rows, headers=["i", "half", "name", "jp"], tablefmt="grid"),
                range(32),
            )
        )
    assert all(o == want for o in outs)


def test_large_table() -> None:
    rows = [[i, i / 7, f"name {i}", i % 2 == 0] for i in range(10_000)]
    out = tabulate(rows, headers=["i", "x", "name", "even"], tablefmt="pipe")
    assert out.count("\n") == 10_001


#: A child that draws a table of `n` rows (`[[123, "abc", "4.5"]] * n`) in Rust. "dry" times the
#: call. "threads" runs a thread that wakes every millisecond and records its longest wait
#: during the call. "sigint" waits for SIGINT from the parent: row 1 is an iterable whose
#: `__iter__` (Python code that the native part runs while it reads the rows) writes "started",
#: and the format's last line, the only other Python code before the call ends, records that
#: the table reached its end.
_LONG_CALL = r"""
import os, signal, sys, threading, time
import pyroid.tabulate as pt

mode, n = sys.argv[1], int(sys.argv[2])
reached_end = []
def last_line(widths, aligns):
    reached_end.append(1)
    return "-"
row = pt.DataRow("", " ", "")
fmt = pt.TableFormat(None, None, None, last_line, row, row, 0, None)
cells = [123, "abc", "4.5"]
rows = [cells] * n

if mode == "dry":
    t0 = time.perf_counter()
    pt.tabulate(rows, tablefmt=fmt)
    print("took", time.perf_counter() - t0)
elif mode == "sigint":
    class Started:
        def __iter__(self):
            os.write(1, b"started\n")
            return iter(cells)
    rows[1] = Started()
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        pt.tabulate(rows, tablefmt=fmt)
        print("finished", 1)
    except KeyboardInterrupt as e:
        os.write(1, b"interrupted\n")
        tb = e.__traceback__
        while tb.tb_next is not None:
            tb = tb.tb_next
        print("raised_in", tb.tb_frame.f_code.co_name)
        print("reached_end", bool(reached_end))
else:
    stop, gaps = [], []
    def count():
        last = time.perf_counter()
        while not stop:
            time.sleep(0.001)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now
    t = threading.Thread(target=count)
    t.start()
    time.sleep(0.05)
    t0 = time.perf_counter()
    pt.tabulate(rows, tablefmt=fmt)
    took = time.perf_counter() - t0
    stop.append(1)
    t.join()
    print("took", took)
    print("longest_gap", max(gaps))
print("after", pt.tabulate([[1]]) == "-\n1\n-")
"""


def _long_call(mode: str, rows: int) -> dict[str, str]:
    out = subprocess.run(
        [sys.executable, "-c", _LONG_CALL, mode, str(rows)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    lines = dict(line.split(" ", 1) for line in out.stdout.splitlines())
    assert lines.pop("after") == "True", out.stdout
    return lines


def _rows_for(seconds: float) -> int:
    """How many rows make `_LONG_CALL` last about `seconds` here, from a dry run."""
    n = 100_000
    took = float(_long_call("dry", n)["took"])
    # At most 4M rows (about 1.2 GB at the call's peak).
    return max(n, min(4_000_000, int(n * seconds / took)))


@pytest.mark.thread_unsafe  # a child process
@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT is delivered differently on Windows")
def test_ctrl_c_stops_a_long_table() -> None:
    with subprocess.Popen(
        [sys.executable, "-c", _LONG_CALL, "sigint", str(_rows_for(1.5))],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as proc:
        assert proc.stdout is not None
        assert proc.stderr is not None
        # The native part is reading the rows: nearly all of the call is still to come.
        assert proc.stdout.readline() == "started\n"
        time.sleep(0.02)  # well after `__iter__` has returned
        sent = time.perf_counter()
        proc.send_signal(signal.SIGINT)
        line = proc.stdout.readline()
        waited = time.perf_counter() - sent
        # Not communicate(): it reads the pipe itself and misses what readline() buffered.
        rest, err = proc.stdout.read(), proc.stderr.read()
    assert proc.returncode == 0, err
    assert line == "interrupted\n", (line, rest, err)
    got = dict(ln.split(" ", 1) for ln in rest.splitlines())
    # Stopped by the call's own signal check, before its end. A call that never checks raises
    # KeyboardInterrupt in the first Python code it runs after the signal: `last_line`.
    assert got["raised_in"] not in ("__iter__", "last_line"), got
    assert got["reached_end"] == "False", got
    assert got["after"] == "True", got
    assert waited < 0.25, waited


@pytest.mark.thread_unsafe  # a child process with threads of its own
def test_a_long_table_lets_other_threads_run() -> None:
    got = _long_call("threads", _rows_for(2.0))
    took, gap = float(got["took"]), float(got["longest_gap"])
    # The other thread wakes every millisecond and waits about 7 ms at a time for the GIL (the
    # interpreter's 5 ms switch interval, then the call's next pause). A step of the call that
    # kept the GIL for a tenth of the call would show.
    assert gap < min(0.25, took / 10), got


@pytest.mark.thread_unsafe  # a child process
def test_command_line_non_utf8_and_an_old_tabulate(tmp_path: Path) -> None:
    # Text read with surrogateescape takes the slow path, which prints what tabulate prints and
    # nothing on stderr; an installed tabulate 0.9.0 is not used.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8:surrogateescape"}
    env["PYTHONPATH"] = str(_old_tabulate(tmp_path))
    out = subprocess.run(
        [sys.executable, "-m", "pyroid.tabulate"],
        input=b"caf\xe9 1\nx 2\n",
        capture_output=True,
        env=env,
        check=True,
    )
    assert out.stdout.replace(b"\r\n", b"\n") == b"----  -\ncaf\xe9  1\nx     2\n----  -\n"
    assert out.stderr == b""


@pytest.mark.thread_unsafe  # a child process
def test_command_line(tmp_path: Path) -> None:
    src = tmp_path / "t.txt"
    src.write_text("a b\n1 2.5\n", encoding="utf-8")
    out = subprocess.run(
        [sys.executable, "-m", "pyroid.tabulate", "-1", "-f", "grid", str(src)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out == "+-----+-----+\n|   a |   b |\n+=====+=====+\n|   1 | 2.5 |\n+-----+-----+\n"
