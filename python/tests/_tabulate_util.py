# ruff: noqa: RUF001  (full-width digits and line separators on purpose)
"""Hypothesis strategies and outcome comparison for the tabulate parity tests.

A generated case is a recipe (plain data), turned into fresh Python objects for each call, so
generators, dataclasses and numpy arrays can be used and nothing one call does can leak into
the other.
"""

from __future__ import annotations

import dataclasses
import datetime
import importlib.util
import decimal
import fractions
import signal
import threading
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

from hypothesis import strategies as st

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_PANDAS = HAS_NUMPY and importlib.util.find_spec("pandas") is not None

FORMATS = [
    "asciidoc",
    "colon_grid",
    "double_grid",
    "double_outline",
    "fancy_grid",
    "fancy_outline",
    "github",
    "grid",
    "heavy_grid",
    "heavy_outline",
    "html",
    "jira",
    "latex",
    "latex_booktabs",
    "latex_longtable",
    "latex_raw",
    "mediawiki",
    "mixed_grid",
    "mixed_outline",
    "moinmoin",
    "orgtbl",
    "outline",
    "pipe",
    "plain",
    "presto",
    "pretty",
    "psql",
    "rounded_grid",
    "rounded_outline",
    "rst",
    "simple",
    "simple_grid",
    "simple_outline",
    "textile",
    "tsv",
    "unsafehtml",
    "youtrack",
]

ESC = "\x1b"
WORDS = [
    "spam",
    "eggs",
    "Hello World",
    " padded ",
    "\tTab",
    "a-b-c",
    "long-hyphenated-word-here",
    "x",
    "日本語",
    "한국어 텍스트",
    "emoji 🙂",
    "é",
    "ümlaut",
    "مرحبا",
    "a\nb",
    "line1\r\nline2",
    "\n",
    "tail\n",
    "\r",
    "a b",
    "a\x1cb",
    f"{ESC}[31mred{ESC}[0m",
    f"{ESC}[1;32mbold green{ESC}[0m text",
    f"{ESC}]8;;https://example.com{ESC}\\link{ESC}]8;;{ESC}\\",
    f"{ESC}]8;id=1;https://x.org{ESC}\\a link{ESC}]8;;{ESC}\\ and more",
    f"{ESC}[",
    "\x01",
    " \x01 ",
    "True",
    "False",
    "None",
    "--",
    "",
    " ",
    "　",
    "&<>\"'",
    "50%_$#{}~^\\",
    "|",
]
NUMERIC = [
    "1",
    "-2",
    "+3",
    " 12 ",
    "007",
    "1_000",
    "1,000",
    "-1,234,567",
    "1,234.5",
    "1,0000",
    "12,34",
    "1.5",
    ".5",
    "5.",
    "-0.0",
    "1e5",
    "1E-3",
    "1e400",
    "-1e-400",
    "42992e1",
    "inf",
    "-inf",
    "Infinity",
    "nan",
    "NaN",
    "-nan",
    "+inf",
    "1_0.5",
    "1__0",
    "0x10",
    "١٢٣",
    "１２",
    "12\n",
    "1,000\n",
    f"{ESC}[31m42{ESC}[0m",
    f"{ESC}[32m3.14159{ESC}[0m",
    f"{ESC}[31m007{ESC}[0m",
    "9" * 30,
    "1" * 4400,
    "123456789012345678901234567890.5",
]


@dataclass
class Point:
    """A dataclass row."""

    x: Any
    y: Any


class NT(NamedTuple):
    """A named-tuple row."""

    a: Any
    b: Any
    c: Any


def _floats() -> st.SearchStrategy[float]:
    return st.one_of(
        st.floats(allow_nan=True, allow_infinity=True),
        st.sampled_from([0.0, -0.0, 1.0, 0.5, 2.5, 1e16, 1e-5, 0.0001, 123456789.123, 1e300]),
        st.floats(min_value=-1e6, max_value=1e6).map(lambda x: round(x, 3)),
    )


# A cell recipe: (kind, payload); `make_cell` builds the object.
cell_recipes = st.one_of(
    st.tuples(st.just("int"), st.integers(-(10**6), 10**6)),
    st.tuples(st.just("int"), st.integers()),
    st.tuples(st.just("int"), st.sampled_from([2**63, -(2**63) - 1, 10**40, -(10**30)])),
    st.tuples(st.just("float"), _floats()),
    st.tuples(st.just("str"), st.sampled_from(WORDS + NUMERIC)),
    st.tuples(st.just("str"), st.text(max_size=12)),
    st.tuples(
        st.just("str"),
        st.text(alphabet="0123456789.,-+e _", min_size=1, max_size=8),
    ),
    st.tuples(st.just("bool"), st.booleans()),
    st.tuples(st.just("none"), st.none()),
    st.tuples(
        st.just("bytes"), st.sampled_from([b"12", b"1.5", b"abc", b"\xff\xfe", b"", b"True"])
    ),
    st.tuples(st.just("decimal"), st.sampled_from(["1.10", "-0", "NaN", "sNaN", "1e3", "3.14159"])),
    st.tuples(st.just("fraction"), st.tuples(st.integers(-9, 9), st.integers(1, 9))),
    st.tuples(st.just("date"), st.integers(0, 20000)),
    *(
        [
            st.tuples(st.just("npfloat"), _floats()),
            st.tuples(st.just("npint"), st.integers(-(2**31), 2**31)),
            st.tuples(st.just("npuint"), st.integers(0, 2**32)),
            st.tuples(st.just("npbool"), st.booleans()),
            st.tuples(st.just("npf32"), st.floats(width=32, allow_nan=True)),
        ]
        if HAS_NUMPY
        else []
    ),
)


def make_cell(recipe: tuple[str, Any]) -> Any:
    kind, v = recipe
    if kind in ("int", "float", "str", "bool", "none", "bytes"):
        return v
    if kind == "decimal":
        return decimal.Decimal(v)
    if kind == "fraction":
        return fractions.Fraction(*v)
    if kind == "date":
        return datetime.date(2000, 1, 1) + datetime.timedelta(days=v)
    import numpy as np

    if kind == "npfloat":
        return np.float64(v)
    if kind == "npint":
        return np.int64(v)
    if kind == "npuint":
        return np.uint64(v)
    if kind == "npbool":
        return np.bool_(v)
    if kind == "npf32":
        return np.float32(v)
    raise AssertionError(kind)


def _row(ncols: int) -> st.SearchStrategy[Any]:
    return st.one_of(
        st.lists(cell_recipes, min_size=ncols, max_size=ncols),
        st.lists(cell_recipes, min_size=0, max_size=ncols + 1),  # ragged
    )


ALIGNS = ["left", "right", "center", "decimal", None, "global", "same", "", "weird"]

FLOATFMTS = [
    "g",
    ".2f",
    ".3g",
    ".1e",
    ",.2f",
    ".0f",
    "e",
    "%",
    ".1%",
    "10.3f",
    "<8",
    "+g",
    "#g",
    ".3",
    "",
    "z.1f",
    "n",
    "_g",
    "05.1f",
    "x",
    ".2,f",
]
INTFMTS = ["", ",", "_", "d", "04d", ">6", "+", "x", "#x", ".2f", "n", "b"]


@st.composite
def tables(draw: st.DrawFn) -> dict[str, Any]:
    """A table recipe and options."""
    ncols = draw(st.integers(0, 4))
    nrows = draw(st.integers(0, 5))
    rows = draw(st.lists(_row(ncols), min_size=nrows, max_size=nrows))
    sep_at = draw(st.lists(st.integers(0, nrows), max_size=2))
    shapes = ["lists", "lists", "lists", "tuples", "generator", "dicts", "dict_of_lists"]
    shapes += ["dataclass", "namedtuple"]
    if HAS_NUMPY:
        shapes.append("numpy")
    if HAS_PANDAS:
        shapes += ["dataframe", "frame_numeric", "frame_numeric"]
    shape = draw(st.sampled_from(shapes))
    headers = draw(
        st.one_of(
            st.just(()),
            st.just("firstrow"),
            st.just("keys"),
            st.lists(
                st.sampled_from(
                    ["a", "bb", "Header", "数字", " x ", "1", "\x1b[1mH\x1b[0m", "two\nlines", ""]
                ),
                max_size=ncols + 1,
            ),
        )
    )
    opts: dict[str, Any] = {}

    def maybe(name: str, strategy: st.SearchStrategy[Any], one_in: int = 2) -> None:
        if draw(st.integers(1, one_in)) == 1:
            opts[name] = draw(strategy)

    tablefmt = draw(
        st.one_of(
            st.sampled_from(FORMATS), st.sampled_from(["unknown", "simple_separated", "custom_tf"])
        )
    )
    maybe(
        "floatfmt",
        st.one_of(st.sampled_from(FLOATFMTS), st.lists(st.sampled_from(FLOATFMTS), max_size=4)),
    )
    maybe(
        "intfmt",
        st.one_of(st.sampled_from(INTFMTS), st.lists(st.sampled_from(INTFMTS), max_size=4)),
    )
    maybe("numalign", st.sampled_from(ALIGNS))
    maybe("stralign", st.sampled_from(ALIGNS))
    maybe(
        "missingval",
        st.one_of(
            st.sampled_from(["", "?", "N/A", "—", " - "]),
            st.lists(st.sampled_from(["", "?", "-"]), max_size=4),
        ),
    )
    maybe(
        "showindex",
        st.sampled_from(["default", "always", "never", True, False, "range", "short_list"]),
    )
    maybe("disable_numparse", st.one_of(st.booleans(), st.lists(st.integers(0, 3), max_size=2)))
    maybe("colglobalalign", st.sampled_from(ALIGNS))
    maybe(
        "colalign",
        st.one_of(st.lists(st.sampled_from(ALIGNS), max_size=5), st.sampled_from(["lr", "right"])),
    )
    maybe("preserve_whitespace", st.booleans())
    maybe(
        "maxcolwidths",
        st.one_of(
            st.integers(1, 12), st.lists(st.one_of(st.none(), st.integers(1, 10)), max_size=4)
        ),
        8,
    )
    if not can_time_out():
        for k in ("maxcolwidths",):
            opts.pop(k, None)
    maybe("headersglobalalign", st.sampled_from(ALIGNS))
    maybe(
        "headersalign",
        st.one_of(st.lists(st.sampled_from(ALIGNS), max_size=5), st.sampled_from(["lr"])),
    )
    maybe(
        "rowalign",
        st.one_of(
            st.sampled_from(["top", "center", "bottom", "weird"]),
            st.lists(st.sampled_from(["top", "center", "bottom", None]), max_size=6),
        ),
    )
    maybe(
        "maxheadercolwidths",
        st.one_of(st.integers(1, 8), st.lists(st.one_of(st.none(), st.integers(1, 8)), max_size=4)),
        8,
    )
    if not can_time_out():
        opts.pop("maxheadercolwidths", None)
    maybe("break_long_words", st.booleans())
    maybe("break_on_hyphens", st.booleans())
    settings = {
        "MIN_PADDING": draw(st.sampled_from([2, 2, 2, 0, 1, 5])),
        "WIDE_CHARS_MODE": draw(st.sampled_from([True, True, False])),
    }
    return {
        "rows": rows,
        "sep_at": sep_at,
        "shape": shape,
        "headers": headers,
        "tablefmt": tablefmt,
        "opts": opts,
        "settings": settings,
    }


def build_args(case: dict[str, Any], mod: Any) -> tuple[Any, dict[str, Any]]:
    """Fresh (tabular_data, kwargs) for one call against module `mod`."""
    rows = [[make_cell(c) for c in r] for r in case["rows"]]
    shape = case["shape"]
    data: Any
    if shape in ("lists", "tuples", "generator"):
        out: list[Any] = [list(r) if shape != "tuples" else tuple(r) for r in rows]
        for i in sorted(case["sep_at"], reverse=True):
            out.insert(min(i, len(out)), mod.SEPARATING_LINE if i % 2 else [mod.SEPARATING_LINE])
        data = (r for r in out) if shape == "generator" else out
    elif shape == "dicts":
        data = [{f"k{j}": v for j, v in enumerate(r)} for r in rows]
    elif shape == "dict_of_lists":
        ncols = max((len(r) for r in rows), default=0)
        data = {f"c{j}": [r[j] for r in rows if j < len(r)] for j in range(ncols)}
    elif shape == "dataclass":
        data = [Point(r[0] if r else None, r[1] if len(r) > 1 else None) for r in rows]
    elif shape == "namedtuple":
        data = [NT(*[*r, None, None, None][:3]) for r in rows]
    elif shape == "numpy":
        import numpy as np

        ncols = max((len(r) for r in rows), default=0)
        data = np.array([(list(r) + [None] * ncols)[:ncols] for r in rows], dtype=object)
        if data.ndim != 2:
            data = [list(r) for r in rows]
    elif shape == "dataframe":
        import pandas as pd

        ncols = max((len(r) for r in rows), default=0)
        data = pd.DataFrame([(list(r) + [None] * ncols)[:ncols] for r in rows])
    elif shape == "frame_numeric":
        import pandas as pd

        ncols = max((len(r) for r in rows), default=0)

        def num(v: Any) -> Any:
            if type(v) is float or (type(v) is int and abs(v) < 2**62):
                return v
            return 0.5 if isinstance(v, float) else 7

        table = [[num(v) for v in (list(r) + [0] * ncols)[:ncols]] for r in rows]
        data = pd.DataFrame(
            table, index=pd.Index(range(10, 10 + len(table)), name="idx") if len(rows) % 2 else None
        )
    else:
        raise AssertionError(shape)
    kwargs = dict(case["opts"])
    headers = case["headers"]
    kwargs["headers"] = list(headers) if isinstance(headers, list) else headers
    if kwargs.get("showindex") == "range":
        kwargs["showindex"] = range(100, 100 + len(rows) + len(case["sep_at"]))
    elif kwargs.get("showindex") == "short_list":
        kwargs["showindex"] = ["i"] * max(0, len(rows) - 1)
    fmt = case["tablefmt"]
    if fmt == "simple_separated":
        fmt = mod.simple_separated_format(";")
    elif fmt == "custom_tf":
        fmt = mod.TableFormat(
            lineabove=mod.Line("<", "~", "+", ">"),
            linebelowheader=mod.Line("[", "=", "|", "]"),
            linebetweenrows=None,
            linebelow=mod.Line("<", "~", "+", ">"),
            headerrow=mod.DataRow("{", "!", "}"),
            datarow=mod.DataRow("(", "|", ")"),
            padding=1,
            with_header_hide=["linebelow"],
        )
    kwargs["tablefmt"] = fmt
    return data, kwargs


@dataclasses.dataclass
class Outcome:
    """The result of one call: text and its type, or the exception; and the warnings."""

    kind: str
    value: str
    warnings: list[str]

    def __str__(self) -> str:
        return f"{self.kind}: {self.value!r} warnings={self.warnings}"


class HangError(Exception):
    """The call ran past its time limit."""


def _alarm(signum: int, frame: Any) -> None:
    raise HangError


def can_time_out() -> bool:
    """Can `outcome` stop a call that hangs (SIGALRM, main thread)?"""
    return hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()


def outcome(
    fn: Callable[[], Any], ignore: tuple[type[Warning], ...] = (), timeout: float = 0
) -> Outcome:
    """Run `fn`; with `timeout`, raise `HangError` if it runs longer (tabulate's text wrapper loops
    forever when a column is narrower than one wide character)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        old = None
        if timeout and can_time_out():
            old = signal.signal(signal.SIGALRM, _alarm)
            signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            r = fn()
        except HangError:
            raise
        except Exception as e:
            out = Outcome(type(e).__name__, str(e), [])
        else:
            out = Outcome(f"result:{type(r).__name__}", r, [])
        finally:
            if old is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old)
    out.warnings = [
        f"{x.category.__name__}: {x.message}" for x in w if not issubclass(x.category, ignore)
    ]
    return out


def with_settings(mod: Any, settings: dict[str, Any], fn: Callable[[], Any]) -> Callable[[], Any]:
    def run() -> Any:
        saved = {k: getattr(mod, k) for k in settings}
        try:
            for k, v in settings.items():
                setattr(mod, k, v)
            return fn()
        finally:
            for k, v in saved.items():
                setattr(mod, k, v)

    return run
