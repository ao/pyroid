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
import subprocess
import sys
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


@pytest.mark.thread_unsafe  # warnings, sys.modules
def test_fallback_without_tabulate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tabulate", None)
    with pytest.raises(pyroid.PyroidError, match="tabulate is not installed"):
        tabulate([["\ud800"]])
    with pytest.raises(pyroid.PyroidError, match="alignment that is not a str"):
        tabulate([[1]], colalign=[3])  # type: ignore[list-item]


@pytest.mark.thread_unsafe  # warnings
def test_fallbacks_warn_and_match_tabulate() -> None:
    real = _real_tabulate()
    pt._WARNED.clear()
    with pytest.warns(pyroid.FallbackWarning, match="lone surrogates"):
        got = tabulate((r for r in [["\ud800", 1], ["x", 2]]), headers=["a", "b"])
    assert got == real.tabulate([["\ud800", 1], ["x", 2]], headers=["a", "b"])
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
    assert len([name for name in port if name in ref]) >= 60


def test_internal_helpers_run_natively() -> None:
    assert pt._align_column(["12.3", "1.0", "100"], "decimal") == [" 12.3", "  1.0", "100  "]
    assert pt._align_column(["a", "bb"], "right", 4) == ["   a", "  bb"]
    assert pt._align_cell_veritically(["x"], 3, 2, "bottom") == ["  ", "  ", "x"]


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
