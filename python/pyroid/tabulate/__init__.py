# Derived from tabulate 0.10.0 (MIT), © 2011-2020 Sergey Astanin and contributors; modified.
"""A drop-in, Rust-backed replacement for tabulate 0.10.0 (``import pyroid.tabulate``).

:func:`tabulate` has tabulate's signature and returns the same text. Column types, number
formatting, alignment and drawing run in Rust; input normalisation and ``maxcolwidths``
wrapping run tabulate's own code. The few calls the Rust code does not take (lone surrogates
in the text, alignments that are not strings, unusual custom ``TableFormat`` fields) run
tabulate 0.10.0's own Python code, which pyroid carries a copy of, with a one-time
:class:`pyroid.FallbackWarning`; tabulate itself never needs to be installed. The full list
is in the guide: https://ataiva.com/open-source/pyroid/docs/guides/tabulate/
"""

from __future__ import annotations

import re
import sys
import threading
import types
import unicodedata
import warnings
from collections.abc import Callable, Iterable, Sized
from typing import Any

from pyroid._pyroid import FallbackWarning
from pyroid._pyroid import tabulate as _native
from pyroid.tabulate import _port
from pyroid.tabulate._port import (
    _BREAK_LONG_WORDS,
    _BREAK_ON_HYPHENS,
    _DEFAULT_ALIGN,
    _DEFAULT_FLOATFMT,
    _DEFAULT_INTFMT,
    _DEFAULT_MISSINGVAL,
    LATEX_ESCAPE_RULES,
    SEPARATING_LINE,
    DataRow,
    JupyterHTMLStr,
    Line,
    TableFormat,
    _afterpoint,
    _align_cell_veritically,
    _align_column,
    _align_header,
    _ansi_codes,
    _bool,
    _column_type,
    _CustomTextWrap,
    _expand_iterable,
    _expand_numparse,
    _format,
    _format_table,
    _html_begin_table_without_header,
    _is_file,
    _is_separating_line,
    _isnumber,
    _more_generic,
    _multiline_width,
    _normalize_tabular_data,
    _prepend_row_index,
    _reinsert_separating_lines,
    _remove_separating_lines,
    _strip_ansi,
    _table_formats,
    _to_str,
    _type,
    _visible_width,
    _wrap_text_to_colwidths,
    multiline_formats,
    simple_separated_format,
    tabulate_formats,
    wcwidth,
)

__all__ = ["tabulate", "tabulate_formats", "simple_separated_format"]  # noqa: RUF022 (tabulate's order)

#: The tabulate release whose output this module reproduces.
TABULATE_VERSION = "0.10.0"
__version__ = TABULATE_VERSION

# minimum extra space in headers
MIN_PADDING = 2

# Whether or not to preserve leading/trailing whitespace in data (unused since tabulate 0.10.0:
# pass preserve_whitespace=True instead).
PRESERVE_WHITESPACE = False

# if True, enable wide-character (CJK) support
WIDE_CHARS_MODE = wcwidth is not None

_LINE_NAMES = ("lineabove", "linebelowheader", "linebetweenrows", "linebelow")

# The built-in formats as they were at import: `_table_formats` may be edited by users.
_BUILTIN: dict[int, tuple[str, Any]] = {id(f): (name, f) for name, f in _table_formats.items()}


def _native_callables() -> tuple[dict[int, tuple[str, Any]], dict[int, tuple[str, Any]]]:
    """The callables of the built-in formats, mapped to the Rust side's names for them."""
    f = _table_formats
    lines = {
        "pipe": f["pipe"].lineabove,
        "grid_colons": f["colon_grid"].linebelowheader,
        "html_begin": f["html"].lineabove,
        "latex_begin": f["latex"].lineabove,
        "latex_booktabs": f["latex_booktabs"].lineabove,
        "latex_longtable": f["latex_longtable"].lineabove,
        "asciidoc_line": f["asciidoc"].lineabove,
    }
    rows = {
        "mediawiki_header": f["mediawiki"].headerrow,
        "mediawiki": f["mediawiki"].datarow,
        "moin_header": f["moinmoin"].headerrow,
        "moin": f["moinmoin"].datarow,
        "html_th": f["html"].headerrow,
        "html_td": f["html"].datarow,
        "unsafehtml_th": f["unsafehtml"].headerrow,
        "unsafehtml_td": f["unsafehtml"].datarow,
        "latex_row": f["latex"].datarow,
        "latex_raw_row": f["latex_raw"].datarow,
        "textile": f["textile"].datarow,
        "asciidoc_header": f["asciidoc"].headerrow,
        "asciidoc_row": f["asciidoc"].datarow,
    }
    return (
        {id(v): (k, v) for k, v in lines.items()},
        {id(v): (k, v) for k, v in rows.items()},
    )


_LINE_NATIVE, _ROW_NATIVE = _native_callables()


class _UnsupportedError(Exception):
    """A format or option the Rust side cannot draw (the call goes to tabulate)."""


def _describe_field(f: Any, n: int, native: dict[int, tuple[str, Any]]) -> Any:
    if not f:
        return None
    if callable(f):
        known = native.get(id(f))
        if known is not None and known[1] is f:
            return ("native", known[0])
        return ("call", f)
    parts = tuple(f)
    if len(parts) != n or not all(type(p) is str for p in parts):
        raise _UnsupportedError("a TableFormat line or row that is not strings or a callable")
    return ("plain", *parts)


def _describe(fmt: Any) -> Any:
    """How the Rust side draws `fmt` (a ``TableFormat``)."""
    builtin = _BUILTIN.get(id(fmt))
    if builtin is not None and builtin[1] is fmt:
        return ("builtin", builtin[0])
    try:
        fields = [_describe_field(f, 4, _LINE_NATIVE) for f in fmt[:4]]
        fields += [_describe_field(f, 3, _ROW_NATIVE) for f in fmt[4:6]]
        padding = fmt.padding
        if type(padding) is not int:
            raise _UnsupportedError("a TableFormat padding that is not an int")
        whh = fmt.with_header_hide
        hide = [bool(name in whh) for name in _LINE_NAMES] if whh else [False] * 4
    except _UnsupportedError:
        raise
    except Exception as e:  # odd objects: tabulate's own code decides what happens
        raise _UnsupportedError(f"a TableFormat it cannot read ({type(e).__name__})") from None
    return ("custom", *fields, padding, hide)


def _table_format_types() -> tuple[type, ...]:
    """Pyroid's ``TableFormat``, and the real tabulate's if it is loaded."""
    real = sys.modules.get("tabulate")
    if getattr(real, "__pyroid__", False):
        real = getattr(real, "__wrapped__", None)
    other = getattr(real, "TableFormat", None)
    if isinstance(other, type) and other is not TableFormat:
        return (TableFormat, other)
    return (TableFormat,)


def _plain_str(s: str) -> bool:
    """`s` has no lone surrogates (the Rust side cannot hold them)."""
    if s.isascii():
        return True
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _simple_align(a: Any) -> bool:
    return a is None or (type(a) is str and _plain_str(a)) or type(a) is bool


_unicode_ok = tuple(int(x) for x in unicodedata.unidata_version.split(".")) <= tuple(
    _native.UNICODE_VERSION
)

_numpy_cache: tuple[Any, list[tuple[type, bool]]] | None = None

_NUMPY_SCALARS = (
    "float64",
    "float32",
    "float16",
    "int8",
    "int16",
    "int32",
    "int64",
    "longlong",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "ulonglong",
    "bool_",
)


def _numpy_types() -> list[tuple[type, bool]]:
    """Numpy scalar types and whether tabulate types them as int (else float).

    tabulate's ``_type`` decides by type alone for these; it is asked once per numpy module.
    """
    global _numpy_cache
    np = sys.modules.get("numpy")
    if np is None:
        return []
    cache = _numpy_cache
    if cache is not None and cache[0] is np:
        return cache[1]
    out: list[tuple[type, bool]] = []
    for name in _NUMPY_SCALARS:
        t = getattr(np, name, None)
        if not isinstance(t, type) or any(t is u for u, _ in out):
            continue
        try:
            kind = _type(t(1))
        except Exception:
            continue
        if kind is int or kind is float:
            out.append((t, kind is int))
    _numpy_cache = (np, out)
    return out


def _escape_empty(val: Any) -> Any:
    """The rst first-column escape of ``_rst_escape_first_column``."""
    if isinstance(val, (str, bytes)) and not val.strip():
        return ".."
    return val


def _ansi_search(s: str) -> bool:
    return _ansi_codes.search(s) is not None


class _Call:
    """What :func:`tabulate` hands the Rust side: settings, and tabulate's option logic."""

    type_fn = staticmethod(_type)
    format_fn = staticmethod(_format)
    strip_ansi_fn = staticmethod(_strip_ansi)
    afterpoint_fn = staticmethod(_afterpoint)
    to_str_fn = staticmethod(_to_str)
    escape_empty_fn = staticmethod(_escape_empty)
    ansi_search_fn = staticmethod(_ansi_search)

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)

    def numparses(self, ncols: int) -> list[bool]:
        return _expand_numparse(self.disable_numparse, ncols)

    def formats(self, ncols: int) -> tuple[list[Any], list[Any], list[Any]]:
        floatfmt, intfmt, missingval = self.floatfmt, self.intfmt, self.missingval
        if isinstance(floatfmt, str):  # old version
            float_formats = ncols * [floatfmt]  # just duplicate the string to use in each column
        else:  # if floatfmt is list, tuple etc we have one per column
            float_formats = list(floatfmt)
            if len(float_formats) < ncols:
                float_formats.extend((ncols - len(float_formats)) * [_DEFAULT_FLOATFMT])
        if isinstance(intfmt, str):  # old version
            int_formats = ncols * [intfmt]  # just duplicate the string to use in each column
        else:  # if intfmt is list, tuple etc we have one per column
            int_formats = list(intfmt)
            if len(int_formats) < ncols:
                int_formats.extend((ncols - len(int_formats)) * [_DEFAULT_INTFMT])
        if isinstance(missingval, str):
            missing_vals = ncols * [missingval]
        else:
            missing_vals = list(missingval)
            if len(missing_vals) < ncols:
                missing_vals.extend((ncols - len(missing_vals)) * [_DEFAULT_MISSINGVAL])
        return float_formats, int_formats, missing_vals

    def aligns(self, codes: list[int]) -> list[Any]:
        # codes: _more_generic's ranks (2 = int, 3 = float)
        if self.colglobalalign is not None:  # if global alignment provided
            aligns = [self.colglobalalign] * len(codes)
        else:  # default
            aligns = [self.numalign if ct in (2, 3) else self.stralign for ct in codes]
        colalign = self.colalign
        if colalign is not None:
            assert isinstance(colalign, Iterable)
            if isinstance(colalign, str):
                warnings.warn(
                    f"As a string, `colalign` is interpreted as {[c for c in colalign]}. "  # noqa: C416
                    f'Did you mean `colglobalalign = "{colalign}"` or `colalign = ("{colalign}",)`?',  # noqa: E501
                    stacklevel=3,
                )
            for idx, align in enumerate(colalign):
                if not idx < len(aligns):
                    break
                elif align != "global":
                    aligns[idx] = align
        return aligns

    def header_aligns(self, aligns: list[Any], n_tcols: int) -> tuple[list[Any], list[Any]]:
        if self.headersglobalalign is not None:  # if global alignment provided
            aligns_headers = [self.headersglobalalign] * n_tcols
        else:  # default
            aligns_headers = aligns or [self.stralign] * len(self.headers)
        headersalign = self.headersalign
        if headersalign is not None:
            assert isinstance(headersalign, Iterable)
            if isinstance(headersalign, str):
                warnings.warn(
                    f"As a string, `headersalign` is interpreted as {[c for c in headersalign]}. "  # noqa: C416
                    f'Did you mean `headersglobalalign = "{headersalign}"` '
                    f'or `headersalign = ("{headersalign}",)`?',
                    stacklevel=3,
                )
            for idx, align in enumerate(headersalign):
                hidx = self.headers_pad + idx
                if not hidx < len(aligns_headers):
                    break
                elif align == "same" and hidx < len(aligns):  # same as column align
                    aligns_headers[hidx] = aligns[hidx]
                elif align != "global":
                    aligns_headers[hidx] = align
        return aligns, aligns_headers

    def rowaligns(self, nrows: int) -> Any:
        rowalign = self.rowalign
        ra_default = rowalign if isinstance(rowalign, str) else None
        return _expand_iterable(rowalign, nrows, ra_default)


# Fallbacks: tabulate 0.10.0's own code (`_port.tabulate`), warning once per reason.
_WARNED: set[str] = set()
_WARN_LOCK = threading.Lock()


def _caller_stacklevel() -> int:
    """The ``stacklevel`` that points ``warnings.warn`` at the first frame outside this module."""
    frame = sys._getframe(1)
    level = 1
    while frame.f_back is not None and frame.f_globals.get("__name__") == __name__:
        frame = frame.f_back
        level += 1
    return level


def _first_time(key: str) -> bool:
    with _WARN_LOCK:
        if key in _WARNED:
            return False
        _WARNED.add(key)
        return True


def _slow_path(reason: str) -> None:
    """Warn, once for `reason`, that the call runs tabulate's Python code."""
    if _first_time(f"tabulate:{reason}"):
        warnings.warn(
            f"pyroid.tabulate.tabulate: {reason} is not supported by the Rust port; "
            f"running tabulate {TABULATE_VERSION}'s own Python code (slow path). "
            "This warning is shown once.",
            FallbackWarning,
            stacklevel=_caller_stacklevel(),
        )


def _run_python(tabular_data: Any, options: dict[str, Any]) -> Any:
    """``tabulate(tabular_data, **options)`` by tabulate 0.10.0's own code (``_port``).

    It is pyroid's copy, not the installed tabulate: that may be another release, and its
    module settings and ``_table_formats`` are its own, not pyroid.tabulate's. A real
    tabulate ``TableFormat`` is passed on as pyroid's class, which the copy recognises.
    """
    fmt = options["tablefmt"]
    foreign = not isinstance(fmt, TableFormat) and isinstance(fmt, _table_format_types())
    if foreign:
        options = dict(options, tablefmt=TableFormat(*fmt))
    text = _port.tabulate(tabular_data, **options)
    if foreign:
        real = _real_module()
        if fmt.lineabove == getattr(real, "_html_begin_table_without_header", None):
            return real.JupyterHTMLStr(text)
    return text


def tabulate(
    tabular_data: Any,
    headers: Any = (),
    tablefmt: Any = "simple",
    floatfmt: Any = _DEFAULT_FLOATFMT,
    intfmt: Any = _DEFAULT_INTFMT,
    numalign: Any = _DEFAULT_ALIGN,
    stralign: Any = _DEFAULT_ALIGN,
    missingval: Any = _DEFAULT_MISSINGVAL,
    showindex: Any = "default",
    disable_numparse: Any = False,
    colglobalalign: Any = None,
    colalign: Any = None,
    preserve_whitespace: Any = False,
    maxcolwidths: Any = None,
    headersglobalalign: Any = None,
    headersalign: Any = None,
    rowalign: Any = None,
    maxheadercolwidths: Any = None,
    break_long_words: Any = _BREAK_LONG_WORDS,
    break_on_hyphens: Any = _BREAK_ON_HYPHENS,
) -> str:
    """Format a fixed width table for pretty printing, like ``tabulate.tabulate``.

    >>> print(tabulate([[1, 2.34], [-56, "8.999"], ["2", "10001"]]))
    ---  ---------
      1      2.34
    -56      8.999
      2  10001
    ---  ---------

    The arguments, defaults and output are tabulate 0.10.0's; see ``__init__.pyi``.
    """
    options: dict[str, Any] = {
        "headers": headers,
        "tablefmt": tablefmt,
        "floatfmt": floatfmt,
        "intfmt": intfmt,
        "numalign": numalign,
        "stralign": stralign,
        "missingval": missingval,
        "showindex": showindex,
        "disable_numparse": disable_numparse,
        "colglobalalign": colglobalalign,
        "colalign": colalign,
        "preserve_whitespace": preserve_whitespace,
        "maxcolwidths": maxcolwidths,
        "headersglobalalign": headersglobalalign,
        "headersalign": headersalign,
        "rowalign": rowalign,
        "maxheadercolwidths": maxheadercolwidths,
        "break_long_words": break_long_words,
        "break_on_hyphens": break_on_hyphens,
    }
    # Alignment lists are read once (tabulate iterates them once too).
    if colalign is not None and isinstance(colalign, Iterable) and not isinstance(colalign, str):
        colalign = options["colalign"] = list(colalign)
    if (
        headersalign is not None
        and isinstance(headersalign, Iterable)
        and not isinstance(headersalign, str)
    ):
        headersalign = options["headersalign"] = list(headersalign)
    tf_types = _table_format_types()
    is_tf = isinstance(tablefmt, tf_types)
    try:
        plain_aligns = (
            type(numalign) is str
            and type(stralign) is str
            and colglobalalign is None
            and headersglobalalign is None
            and colalign is None
            and headersalign is None
        )
        if not plain_aligns and (
            not all(
                _simple_align(a) for a in (numalign, stralign, colglobalalign, headersglobalalign)
            )
            or not all(
                _simple_align(a)
                for seq in (colalign, headersalign)
                if isinstance(seq, list)
                for a in seq
            )
        ):
            raise _UnsupportedError("an alignment that is not a str")
        if type(MIN_PADDING) is not int:
            raise _UnsupportedError("a MIN_PADDING that is not an int")
        fmt_given = _describe(tablefmt) if is_tf else None
    except _UnsupportedError as e:
        _slow_path(str(e))
        return _run_python(tabular_data, options)  # type: ignore[no-any-return]

    if tabular_data is None:
        tabular_data = []

    fast = None
    if maxcolwidths is None and maxheadercolwidths is None:
        fast = _plain_rows(tabular_data, headers, showindex)
        if fast is None:
            fast = _frame_rows(tabular_data, headers, showindex)
    index = None
    if fast is not None:
        # The rows go to Rust as they are; it converts them, adds the index and finds the
        # separators.
        list_of_lists, headers, headers_pad, index = fast
        rows0, headers0 = list_of_lists, headers[headers_pad:]
        normalized = lambda: (_with_index(rows0, index), headers0)  # noqa: E731
        separating_lines = None
    else:
        list_of_lists, headers, headers_pad = _normalize_tabular_data(
            tabular_data, headers, showindex=showindex
        )
        rows1, headers1 = list_of_lists, headers[headers_pad:]
        normalized = lambda: (rows1, headers1)  # noqa: E731
        list_of_lists, separating_lines = _remove_separating_lines(list_of_lists)

    if maxcolwidths is not None:
        if type(maxcolwidths) is tuple:  # Check if tuple, convert to list if so
            maxcolwidths = list(maxcolwidths)
        if len(list_of_lists):  # noqa: SIM108 (tabulate's code)
            num_cols = len(list_of_lists[0])
        else:
            num_cols = 0
        if isinstance(maxcolwidths, int):  # Expand scalar for all columns
            maxcolwidths = _expand_iterable(maxcolwidths, num_cols, maxcolwidths)
        else:  # Ignore col width for any 'trailing' columns
            maxcolwidths = _expand_iterable(maxcolwidths, num_cols, None)

        numparses = _expand_numparse(disable_numparse, num_cols)
        list_of_lists = _wrap_text_to_colwidths(
            list_of_lists,
            maxcolwidths,
            numparses=numparses,
            missingval=missingval,
            break_long_words=break_long_words,
            break_on_hyphens=break_on_hyphens,
        )

    if maxheadercolwidths is not None:
        num_cols = len(list_of_lists[0]) if list_of_lists else len(headers)
        if isinstance(maxheadercolwidths, int):  # Expand scalar for all columns
            maxheadercolwidths = _expand_iterable(maxheadercolwidths, num_cols, maxheadercolwidths)
        else:  # Ignore col width for any 'trailing' columns
            maxheadercolwidths = _expand_iterable(maxheadercolwidths, num_cols, None)

        numparses = _expand_numparse(disable_numparse, num_cols)
        headers = _wrap_text_to_colwidths(
            [headers],
            maxheadercolwidths,
            numparses=numparses,
            missingval=missingval,
            break_long_words=break_long_words,
            break_on_hyphens=break_on_hyphens,
        )[0]

    # PrettyTable formatting does not use any extra padding.
    # Numbers are not parsed and are treated the same as strings for alignment.
    # Check if pretty is the format being used and override the defaults so it
    # does not impact other formats.
    min_padding = MIN_PADDING
    if tablefmt == "pretty":
        min_padding = 0
        disable_numparse = True
        numalign = "center" if numalign == _DEFAULT_ALIGN else numalign
        stralign = "center" if stralign == _DEFAULT_ALIGN else stralign
    else:
        numalign = "decimal" if numalign == _DEFAULT_ALIGN else numalign
        stralign = "left" if stralign == _DEFAULT_ALIGN else stralign

    # 'colon_grid' uses colons in the line beneath the header to represent a column's
    # alignment instead of literally aligning the text differently. Hence,
    # left alignment of the data in the text output is enforced.
    if tablefmt == "colon_grid":
        colglobalalign = "left"
        headersglobalalign = "left"

    # The format for single-line and for multiline cells (tabulate picks one once it knows).
    try:
        if is_tf:
            plain_fmt, multi_fmt, multi_name = tablefmt, None, None
            desc_plain, desc_multi = fmt_given, None
        else:
            plain_fmt = _table_formats.get(tablefmt, _table_formats["simple"])
            multi_name = multiline_formats.get(tablefmt, tablefmt)
            multi_fmt = None
            if tablefmt in multiline_formats:
                multi_fmt = (
                    multi_name
                    if isinstance(multi_name, tf_types)
                    else _table_formats.get(multi_name, _table_formats["simple"])
                )
            desc_plain = _describe(plain_fmt)
            desc_multi = None if multi_fmt is None else _describe(multi_fmt)
    except _UnsupportedError as e:
        return _delegate_normalized(str(e), normalized, options)

    enable_widechars = wcwidth is not None and WIDE_CHARS_MODE
    call = _Call(
        unicode_ok=_unicode_ok,
        max_digits=sys.get_int_max_str_digits(),
        is_rst=tablefmt == "rst",
        fmt_plain=desc_plain,
        fmt_multi=desc_multi,
        multiline_candidate=multi_fmt is not None,
        colon_plain=tablefmt == "colon_grid",
        colon_multi=multi_name == "colon_grid",
        wide=enable_widechars,
        wcswidth=wcwidth.wcswidth if enable_widechars else None,
        numpy_types=_numpy_types(),
        preserve_whitespace=preserve_whitespace,
        min_padding=min_padding,
        disable_numparse=disable_numparse,
        floatfmt=floatfmt,
        intfmt=intfmt,
        missingval=missingval,
        numalign=numalign,
        stralign=stralign,
        colglobalalign=colglobalalign,
        colalign=colalign,
        headersglobalalign=headersglobalalign,
        headersalign=headersalign,
        headers=headers,
        headers_pad=headers_pad,
        rowalign=rowalign,
        raw_rows=fast is not None,
        index=index,
        simple_options=(
            type(disable_numparse) is bool
            and type(floatfmt) is str
            and type(intfmt) is str
            and type(missingval) is str
            and colalign is None
            and headersalign is None
            and (rowalign is None or type(rowalign) is str)
        ),
    )
    text, multiline, empty = _native.render(list_of_lists, headers, separating_lines, call)
    if text is None:
        return _delegate_normalized(multiline, normalized, options)
    if empty:  # a completely empty table
        return ""
    fmt = multi_fmt if multiline else plain_fmt
    if fmt.lineabove == _html_begin_table_without_header:
        return JupyterHTMLStr(text)
    real = _real_module()
    if real is not None and fmt.lineabove == getattr(
        real, "_html_begin_table_without_header", None
    ):
        return real.JupyterHTMLStr(text)  # type: ignore[no-any-return]
    return text  # type: ignore[no-any-return]


def _index_for(showindex: Any, index: Any, rows: list[Any]) -> Any:
    """The index column ``_normalize_tabular_data`` adds (``None``: none; ``False``: give up).

    `index` is the DataFrame's index, or ``None``. Lazy iterables are left to the slow path.
    """
    showindex_is_a_str = type(showindex) in [str, bytes]
    if showindex == "default" and index is not None:
        idx = index
    elif isinstance(showindex, Sized) and not showindex_is_a_str:
        idx = list(showindex)
    elif isinstance(showindex, Iterable) and not showindex_is_a_str:
        return False
    elif showindex == "always" or (_bool(showindex) and not showindex_is_a_str):
        idx = index if index is not None else list(range(len(rows)))
    else:
        return None
    if len(idx) != len(rows):
        raise ValueError(
            "index must be as long as the number of data rows: "
            + f"len(index)={len(idx)} len(rows)={len(rows)}"
        )
    return idx


def _padded_headers(headers: list[str], rows: list[Any], idx: Any) -> tuple[list[str], int]:
    """Tabulate's empty headers for the first columns (the row length after the index)."""
    headers_pad = 0
    if headers and len(rows) > 0:
        first = rows[0]
        n = len(first)
        if idx is not None:
            # tabulate measures its first row after `_prepend_row_index`, which puts
            # SEPARATING_LINE itself (one character) in the place of a separator row. A tuple
            # row is a list by then, so it can be a separator too.
            as_row = list(first) if type(first) is tuple else first
            n = len(SEPARATING_LINE) if _is_separating_line(as_row) else n + 1
        headers_pad = max(0, n - len(headers))
        headers = [""] * headers_pad + headers
    return headers, headers_pad


_ROW_TYPES = (list, tuple, str)


def _plain_rows(tabular_data: Any, headers: Any, showindex: Any) -> Any:
    """``_normalize_tabular_data`` for a plain list of rows, without converting the rows.

    For a ``list`` whose first row is a list, tuple or str, tabulate's normalisation only
    reads the first row (for ``"keys"`` and ``"firstrow"``), turns every row into a list and
    adds an index; the Rust side does those. Returns ``(rows, headers, headers_pad, index)``,
    or ``None`` for any other input.
    """
    if type(tabular_data) is not list:
        return None
    rows = tabular_data
    if rows and type(rows[0]) not in _ROW_TYPES:
        return None
    try:
        bool(headers)
    except ValueError:  # numpy.ndarray, pandas.core.index.Index, ...
        headers = list(headers)
    if headers == "keys":
        headers = list(map(str, range(len(rows[0])))) if rows else []
    elif headers == "firstrow":
        if rows:
            headers = rows[0]
            rows = rows[1:]
            if rows and type(rows[0]) not in _ROW_TYPES:
                return None
        else:
            headers = []
    headers = list(map(str, headers))
    idx = _index_for(showindex, None, rows)
    if idx is False:
        return None
    headers, headers_pad = _padded_headers(headers, rows, idx)
    return rows, headers, headers_pad, idx


def _frame_rows(tabular_data: Any, headers: Any, showindex: Any) -> Any:
    """``_normalize_tabular_data`` for a pandas DataFrame of float64s, signed ints or objects.

    tabulate builds its rows with ``[list(row) for row in df.values]``. For these dtypes
    ``df.values.tolist()`` gives objects tabulate types and formats the same way, and whose
    ``str()`` is the same (``headers="firstrow"``, ``disable_numparse``), so the rows come
    from that. float32 and float16 scalars print fewer digits than the Python floats
    ``tolist()`` makes of them (``1.1``, not ``1.100000023841858``), and tabulate types
    unsigned ints and booleans as floats: those frames are left to the slow path.
    """
    if not (hasattr(tabular_data, "keys") and hasattr(tabular_data, "values")):
        return None
    if callable(tabular_data.values) or not hasattr(tabular_data, "index"):
        return None
    vals = tabular_data.values
    dtype = getattr(vals, "dtype", None)
    kind = getattr(dtype, "kind", None)
    if getattr(vals, "ndim", None) != 2 or kind not in ("f", "i", "O"):
        return None
    if kind == "f" and getattr(dtype, "char", None) != "d":  # float64 only
        return None
    try:
        bool(headers)
    except ValueError:  # numpy.ndarray, pandas.core.index.Index, ...
        headers = list(headers)
    keys = list(tabular_data)
    if showindex in ["default", "always", True] and tabular_data.index.name is not None:
        if isinstance(tabular_data.index.name, list):
            keys[:0] = tabular_data.index.name
        else:
            keys[:0] = [tabular_data.index.name]
    index = list(tabular_data.index)
    rows = vals.tolist()
    if headers == "keys":
        headers = list(map(str, keys))  # headers should be strings
    if headers == "firstrow" and len(rows) > 0:
        headers = [index[0]] + list(rows[0])  # noqa: RUF005 (tabulate's code)
        index = index[1:]
        rows = rows[1:]
    elif headers == "firstrow":
        headers = []
    headers = list(map(str, headers))
    idx = _index_for(showindex, index, rows)
    if idx is False:
        return None
    headers, headers_pad = _padded_headers(headers, rows, idx)
    return rows, headers, headers_pad, idx


def _with_index(rows: list[Any], index: Any) -> list[Any]:
    """`rows` with the index column added as tabulate adds it (for handing a call over)."""
    if index is None:
        return rows
    return _prepend_row_index([r if _is_separating_line(r) else list(r) for r in rows], index)


def _real_module() -> Any:
    real = sys.modules.get("tabulate")
    if getattr(real, "__pyroid__", False):
        real = getattr(real, "__wrapped__", None)
    return real


def _delegate_normalized(
    reason: str, normalized: Callable[[], Any], options: dict[str, Any]
) -> str:
    """Run a call whose input was already normalised by tabulate's Python code.

    The input may have been a generator, so tabulate's code gets the normalised rows and
    headers (which it normalises to themselves) and no index.
    """
    rows, headers = normalized()
    _slow_path(reason)
    return _run_python(rows, dict(options, headers=headers, showindex=False))  # type: ignore[no-any-return]


_USAGE = """\
    Usage: tabulate [options] [FILE ...]

    Pretty-print tabular data.
    See also https://github.com/astanin/python-tabulate

    FILE                      a filename of the file with tabular data;
                              if "-" or missing, read data from stdin.

    Options:

    -h, --help                show this message
    -1, --header              use the first row of data as a table header
    -o FILE, --output FILE    print table to FILE (default: stdout)
    -s REGEXP, --sep REGEXP   use a custom column separator (default: whitespace)
    -F FPFMT, --float FPFMT   floating point number format (default: g)
    -I INTFMT, --int INTFMT   integer point number format (default: "")
    -f FMT, --format FMT      set output table format; supported formats:
                              plain, simple, grid, fancy_grid, pipe, orgtbl,
                              rst, mediawiki, html, latex, latex_raw,
                              latex_booktabs, latex_longtable, tsv
                              (default: simple)
    """


def _main() -> None:
    """Run tabulate's command line (``python -m pyroid.tabulate``)."""
    import getopt
    import textwrap

    usage = textwrap.dedent(_USAGE)
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "h1o:s:F:I:f:",
            [
                "help",
                "header",
                "output=",
                "sep=",
                "float=",
                "int=",
                "colalign=",
                "format=",
            ],
        )
    except getopt.GetoptError as e:
        print(e)  # noqa: T201
        print(usage)  # noqa: T201
        sys.exit(2)
    headers: Any = []
    floatfmt = _DEFAULT_FLOATFMT
    intfmt = _DEFAULT_INTFMT
    colalign = None
    tablefmt = "simple"
    sep = r"\s+"
    outfile = "-"
    for opt, value in opts:
        if opt in ["-1", "--header"]:
            headers = "firstrow"
        elif opt in ["-o", "--output"]:
            outfile = value
        elif opt in ["-F", "--float"]:
            floatfmt = value
        elif opt in ["-I", "--int"]:
            intfmt = value
        elif opt in ["-C", "--colalign"]:
            colalign = value.split()  # type: ignore[assignment]
        elif opt in ["-f", "--format"]:
            if value not in tabulate_formats:
                print("%s is not a supported table format" % value)  # noqa: T201, UP031
                print(usage)  # noqa: T201
                sys.exit(3)
            tablefmt = value
        elif opt in ["-s", "--sep"]:
            sep = value
        elif opt in ["-h", "--help"]:
            print(usage)  # noqa: T201
            sys.exit(0)
    files = [sys.stdin] if not args else args  # noqa: SIM212
    # The slow-path notice is for code that calls tabulate(); tabulate's command line prints
    # nothing on stderr for, say, text read with surrogateescape.
    with (
        warnings.catch_warnings(),
        sys.stdout if outfile == "-" else open(outfile, "w") as out,
    ):
        warnings.simplefilter("ignore", FallbackWarning)
        for f in files:
            if f == "-":
                f = sys.stdin
            if _is_file(f):
                _pprint_file(
                    f,
                    headers=headers,
                    tablefmt=tablefmt,
                    sep=sep,
                    floatfmt=floatfmt,
                    intfmt=intfmt,
                    file=out,
                    colalign=colalign,
                )
            else:
                with open(f) as fobj:  # type: ignore[arg-type]
                    _pprint_file(
                        fobj,
                        headers=headers,
                        tablefmt=tablefmt,
                        sep=sep,
                        floatfmt=floatfmt,
                        intfmt=intfmt,
                        file=out,
                        colalign=colalign,
                    )


def _pprint_file(
    fobject: Any,
    headers: Any,
    tablefmt: Any,
    sep: Any,
    floatfmt: Any,
    intfmt: Any,
    file: Any,
    colalign: Any,
) -> None:
    rows = fobject.readlines()
    table = [re.split(sep, r.rstrip()) for r in rows if r.strip()]
    print(
        tabulate(
            table,
            headers,
            tablefmt,
            floatfmt=floatfmt,
            intfmt=intfmt,
            colalign=colalign,
        ),
        file=file,
    )


# pyroid.patch("tabulate")
# ------------------------
#: The names of tabulate 0.10.0 the ``tabulate`` shim takes from here: all its functions,
#: classes and tables (what else it has, the modules it imports, forwards to the real tabulate
#: when it is installed).
_PATCHED_NAMES: tuple[str, ...] = (
    "DataRow",
    "JupyterHTMLStr",
    "LATEX_ESCAPE_RULES",
    "Line",
    "SEPARATING_LINE",
    "TableFormat",
    "_BREAK_LONG_WORDS",
    "_BREAK_ON_HYPHENS",
    "_CustomTextWrap",
    "_DEFAULT_ALIGN",
    "_DEFAULT_FLOATFMT",
    "_DEFAULT_INTFMT",
    "_DEFAULT_MISSINGVAL",
    "_afterpoint",
    "_align_cell_veritically",
    "_align_column",
    "_align_column_choose_padfn",
    "_align_column_choose_width_fn",
    "_align_column_multiline_width",
    "_align_header",
    "_ansi_codes",
    "_ansi_codes_bytes",
    "_ansi_color_reset_code",
    "_ansi_escape_pat",
    "_append_basic_row",
    "_append_line",
    "_append_multiline_row",
    "_asciidoc_row",
    "_bool",
    "_build_line",
    "_build_row",
    "_build_simple_row",
    "_choose_width_fn",
    "_column_type",
    "_csi",
    "_esc",
    "_expand_iterable",
    "_expand_numparse",
    "_flat_list",
    "_float_with_thousands_separators",
    "_format",
    "_format_table",
    "_grid_line_with_colons",
    "_grid_segment_with_colons",
    "_html_begin_table_without_header",
    "_html_row_with_attrs",
    "_is_file",
    "_is_multiline",
    "_is_separating_line",
    "_is_separating_line_value",
    "_isbool",
    "_isconvertible",
    "_isint",
    "_isnumber",
    "_isnumber_with_thousands_separator",
    "_latex_line_begin_tabular",
    "_latex_row",
    "_main",
    "_mediawiki_row_with_attrs",
    "_moin_row_with_attrs",
    "_more_generic",
    "_multiline_codes",
    "_multiline_codes_bytes",
    "_multiline_width",
    "_normalize_tabular_data",
    "_osc",
    "_pad_row",
    "_padboth",
    "_padleft",
    "_padnone",
    "_padright",
    "_pipe_line_with_colons",
    "_pipe_segment_with_colons",
    "_pprint_file",
    "_prepend_row_index",
    "_reinsert_separating_lines",
    "_remove_separating_lines",
    "_rst_escape_first_column",
    "_st",
    "_strip_ansi",
    "_table_formats",
    "_textile_row_with_attrs",
    "_to_str",
    "_type",
    "_visible_width",
    "_wrap_text_to_colwidths",
    "multiline_formats",
    "simple_separated_format",
    "tabulate",
    "tabulate_formats",
)

#: Module settings ``tabulate()`` reads on each call: writes on the shim land here too, and
#: writes here reach the copied tabulate code in ``_port`` (`_Module`).
_SETTINGS = ("MIN_PADDING", "PRESERVE_WHITESPACE", "WIDE_CHARS_MODE", "wcwidth")


class _Module(types.ModuleType):
    """This module's type: a setting written on it is written on ``_port`` too.

    The Rust path reads the settings here, tabulate's copied code (the fallback, the text
    wrapper) in ``_port``; both must see the same values.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name in _SETTINGS:
            setattr(_port, name, value)

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)
        if name in _SETTINGS and name in vars(_port):
            delattr(_port, name)


sys.modules[__name__].__class__ = _Module

_shim_classes: dict[type, type] = {}


# tabulate's private helpers not imported above, for code that reaches for them
# (``pyroid.tabulate._padleft``): tabulate's own code, from `_port`.
for _name in _PATCHED_NAMES:
    if _name not in globals():
        globals()[_name] = getattr(_port, _name)
del _name


def _patched_names() -> dict[str, Any]:
    """The objects ``pyroid.patch("tabulate")`` puts on its ``tabulate`` shim."""
    here = globals()
    return {n: here[n] for n in _PATCHED_NAMES}


def _share_settings(shim: Any, real: Any) -> None:
    """Make `shim` (the patched ``tabulate``) show this module's settings and pass writes on.

    Settings changed on the real module before the patch (``tabulate.MIN_PADDING = 0``) are
    carried over first.
    """
    module = sys.modules[__name__]
    if real is not None:
        defaults = {
            "MIN_PADDING": 2,
            "PRESERVE_WHITESPACE": False,
            "WIDE_CHARS_MODE": getattr(real, "wcwidth", None) is not None,
        }
        for name, default in defaults.items():
            value = getattr(real, name, default)
            if value != default:
                setattr(module, name, value)
    for name in _SETTINGS:
        shim.__dict__[name] = getattr(module, name)
    base = type(shim)
    cls = _shim_classes.get(base)
    if cls is None:

        class _SettingsShim(base):  # type: ignore[valid-type, misc]
            def __setattr__(self, name: str, value: Any) -> None:
                if name in _SETTINGS:
                    setattr(module, name, value)
                super().__setattr__(name, value)

        _SettingsShim.__name__ = _SettingsShim.__qualname__ = base.__name__
        cls = _shim_classes.setdefault(base, _SettingsShim)
    shim.__class__ = cls


# tabulate's other module-level names, for code (and tests) that reach for them.
_ = (
    DataRow,
    LATEX_ESCAPE_RULES,
    Line,
    SEPARATING_LINE,
    _CustomTextWrap,
    _align_cell_veritically,
    _align_column,
    _align_header,
    _column_type,
    _format_table,
    _isnumber,
    _more_generic,
    _multiline_width,
    _port,
    _reinsert_separating_lines,
    _visible_width,
)
del _
