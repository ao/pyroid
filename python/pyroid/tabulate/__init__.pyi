# Derived from tabulate 0.10.0 (MIT), © 2011-2020 Sergey Astanin and contributors; modified.
"""A drop-in, Rust-backed replacement for tabulate 0.10.0.

```python
from pyroid.tabulate import tabulate

print(tabulate([["spam", 41.9999], ["eggs", "451.0"]], headers=["item", "qty"], tablefmt="github"))
```

[`tabulate`][pyroid.tabulate.tabulate] has tabulate's signature and returns the same text:
column types, number formatting, alignment and drawing run in Rust, while input
normalisation and `maxcolwidths` wrapping run tabulate's own code. The module also has
tabulate's other names (`tabulate_formats`, `TableFormat`, `Line`, `DataRow`,
`SEPARATING_LINE`, `simple_separated_format`, `MIN_PADDING`, `WIDE_CHARS_MODE`, ...), and
`python -m pyroid.tabulate` is tabulate's command line.

The few calls the Rust code does not take run tabulate 0.10.0's own Python code, which pyroid
carries a copy of, with a one-time [`pyroid.FallbackWarning`][pyroid.FallbackWarning]: text
holding lone surrogates, alignments that are not strings (`None` and booleans are fine), custom
`TableFormat` fields that are neither strings nor callables, and the others listed in the
[guide](../guides/tabulate.md#what-runs-where). tabulate itself never needs to be installed.
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Final, NamedTuple

__all__ = ["TABULATE_VERSION", "simple_separated_format", "tabulate", "tabulate_formats"]

TABULATE_VERSION: Final[str]
"""The tabulate release whose output this module reproduces (`"0.10.0"`)."""

__version__: Final[str]
"""Same as [`TABULATE_VERSION`][pyroid.tabulate.TABULATE_VERSION]."""

MIN_PADDING: int
"""Extra space around headers (2); read on each call, as in tabulate."""

PRESERVE_WHITESPACE: bool
"""Kept for compatibility; unused since tabulate 0.10.0 (pass `preserve_whitespace=True`)."""

WIDE_CHARS_MODE: bool
"""Measure text with `wcwidth.wcswidth` (true when wcwidth is installed); read on each call."""

SEPARATING_LINE: Final[str]
"""A row equal to this (or starting with it) is drawn as a horizontal line."""

tabulate_formats: list[str]
"""The names of the built-in table formats, sorted."""

multiline_formats: dict[str, str]
"""The formats that fold multiline cells into several lines."""

LATEX_ESCAPE_RULES: dict[str, str]
"""The character escapes of the `latex` formats."""

class Line(NamedTuple):
    """A horizontal line of a `TableFormat`."""

    begin: str
    hline: str
    sep: str
    end: str

class DataRow(NamedTuple):
    """A row of a `TableFormat`."""

    begin: str
    sep: str
    end: str

class TableFormat(NamedTuple):
    """A table format: lines and rows (each a `Line`/`DataRow`, a callable or `None`)."""

    lineabove: Line | Callable[[list[int], list[Any]], str] | None
    linebelowheader: Line | Callable[[list[int], list[Any]], str] | None
    linebetweenrows: Line | Callable[[list[int], list[Any]], str] | None
    linebelow: Line | Callable[[list[int], list[Any]], str] | None
    headerrow: DataRow | Callable[[list[str], list[int], list[Any]], str] | None
    datarow: DataRow | Callable[[list[str], list[int], list[Any]], str] | None
    padding: int
    with_header_hide: list[str] | None

class JupyterHTMLStr(str):
    """The `html` formats' result: a `str` that Jupyter displays as HTML."""

    @property
    def str(self) -> str:
        """The text itself."""

def simple_separated_format(separator: str) -> TableFormat:
    """Return a format with columns separated by `separator` and nothing else.

    Args:
        separator: The column separator.

    Returns:
        A `TableFormat`, as `tabulate.simple_separated_format` builds it.
    """

def tabulate(
    tabular_data: Any,
    headers: str | Iterable[Any] | Mapping[Any, Any] = (),
    tablefmt: str | TableFormat = "simple",
    floatfmt: str | Iterable[str] = "g",
    intfmt: str | Iterable[str] = "",
    numalign: str | None = "default",
    stralign: str | None = "default",
    missingval: str | Iterable[str] = "",
    showindex: str | bool | Iterable[Any] = "default",
    disable_numparse: bool | Iterable[int] = False,
    colglobalalign: str | None = None,
    colalign: Iterable[str | None] | None = None,
    preserve_whitespace: bool = False,
    maxcolwidths: int | Iterable[int | None] | None = None,
    headersglobalalign: str | None = None,
    headersalign: Iterable[str | None] | None = None,
    rowalign: str | Iterable[str | None] | None = None,
    maxheadercolwidths: int | Iterable[int | None] | None = None,
    break_long_words: bool = True,
    break_on_hyphens: bool = True,
) -> str:
    """Format a table as text, exactly as `tabulate.tabulate` 0.10.0 does.

    ```python
    >>> print(tabulate([[1, 2.34], [-56, "8.999"], ["2", "10001"]]))
    ---  ---------
      1      2.34
    -56      8.999
      2  10001
    ---  ---------
    ```

    Args:
        tabular_data: Rows: a list (or other iterable) of lists or tuples, a list of dicts,
            dataclasses or named tuples, a dict of columns, a 2-D numpy array or record
            array, a pandas DataFrame (its index is shown by default), or a DB-API cursor.
        headers: Column headers: a list (aligned to the last columns if shorter),
            `"firstrow"`, `"keys"` (dict keys, field names or column indices), or a dict
            mapping the keys of dict rows to headers.
        tablefmt: A name from `tabulate_formats` or a `TableFormat`; unknown names draw
            `"simple"`.
        floatfmt: A format spec for float columns (`"g"`, `".2f"`, ...), or one per column.
        intfmt: A format spec for int columns (`","`), or one per column.
        numalign: `"right"`, `"center"`, `"left"`, `"decimal"` (the default) or `None` for
            number columns.
        stralign: `"left"` (the default), `"right"`, `"center"` or `None` for text columns.
        missingval: The text for `None` cells, or one per column.
        showindex: `"default"` (a DataFrame's index), `"always"` or `True`, `"never"` or
            `False`, or the index values to show.
        disable_numparse: Treat numbers as text: `True`, or a list of column indices.
        colglobalalign: One alignment for every column (before `colalign`).
        colalign: Per-column alignments; `"global"` keeps the global one.
        preserve_whitespace: Keep leading and trailing whitespace in cells.
        maxcolwidths: Wrap text columns at this width (one int, or one per column).
        headersglobalalign: One alignment for every header.
        headersalign: Per-header alignments; `"same"` follows the column, `"global"` keeps
            the global one.
        rowalign: Vertical alignment of multiline rows (`"top"`, `"center"`, `"bottom"`),
            or one per row.
        maxheadercolwidths: Wrap headers at this width (one int, or one per column).
        break_long_words: Break words longer than the width when wrapping.
        break_on_hyphens: Break wrapped text at hyphens.

    Returns:
        The table; a `JupyterHTMLStr` for the `html` and `unsafehtml` formats.

    Raises:
        TypeError: The data is not two-dimensional, among tabulate's other errors.
        ValueError: An index of the wrong length, among tabulate's other errors.
    """

def _main() -> None:
    """Tabulate's command line (`python -m pyroid.tabulate`)."""
