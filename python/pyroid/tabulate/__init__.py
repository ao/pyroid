# Derived from tabulate 0.10.0 (MIT), © 2011-2020 Sergey Astanin and contributors; modified.
"""A drop-in, Rust-backed replacement for tabulate 0.10.0 (``import pyroid.tabulate``).

:func:`tabulate` has tabulate's signature and returns the same text. Column types, number
formatting, alignment and drawing run in Rust; input normalisation and ``maxcolwidths``
wrapping run tabulate's own code. Calls the port does not handle (text with lone surrogates,
alignments that are not strings, custom ``TableFormat`` fields that are neither strings nor
callables) go to the real tabulate with a one-time :class:`pyroid.FallbackWarning`, or raise
:class:`pyroid.PyroidError` if it is not installed. See ``__init__.pyi``.
"""

from __future__ import annotations

import re
import sys
import threading
import unicodedata
import warnings
from collections.abc import Iterable
from typing import Any

from pyroid._pyroid import FallbackWarning, PyroidError
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
    _ansi_codes,
    _column_type,
    _CustomTextWrap,
    _expand_iterable,
    _expand_numparse,
    _format,
    _html_begin_table_without_header,
    _is_file,
    _isnumber,
    _more_generic,
    _normalize_tabular_data,
    _reinsert_separating_lines,
    _remove_separating_lines,
    _strip_ansi,
    _table_formats,
    _to_str,
    _type,
    _wrap_text_to_colwidths,
    multiline_formats,
    simple_separated_format,
    tabulate_formats,
    wcwidth,
)

__all__ = ["TABULATE_VERSION", "simple_separated_format", "tabulate", "tabulate_formats"]

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


def _simple_align(a: Any) -> bool:
    return a is None or type(a) is str or type(a) is bool


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


# Fallbacks: the real tabulate, warning once per reason.
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


def _real_tabulate(reason: str) -> Any:
    """The real tabulate module, warning once for `reason`, or PyroidError."""
    try:
        import tabulate as real
    except ImportError:
        real = None
    # Under a pyroid patch shim, fall back to the real module it wraps, never to the shim.
    if getattr(real, "__pyroid__", False):
        real = real.__wrapped__
    if real is None:
        raise PyroidError(
            f"pyroid.tabulate.tabulate: {reason} is not supported by pyroid's Rust port, and "
            f"tabulate is not installed to fall back to (pip install tabulate=={TABULATE_VERSION})"
        )
    if _first_time("version"):
        from importlib import metadata

        try:
            version = metadata.version("tabulate")
        except metadata.PackageNotFoundError:
            version = None
        if version is not None and version != TABULATE_VERSION:
            warnings.warn(
                f"pyroid.tabulate is falling back to tabulate {version}, not {TABULATE_VERSION}",
                RuntimeWarning,
                stacklevel=_caller_stacklevel(),
            )
    if _first_time(f"tabulate:{reason}"):
        warnings.warn(
            f"pyroid.tabulate.tabulate: {reason} is not supported by the Rust port; "
            "delegating to tabulate (slow path). This warning is shown once.",
            FallbackWarning,
            stacklevel=_caller_stacklevel(),
        )
    return real


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
    if isinstance(colalign, Iterable) and not isinstance(colalign, str):
        colalign = options["colalign"] = list(colalign)
    if isinstance(headersalign, Iterable) and not isinstance(headersalign, str):
        headersalign = options["headersalign"] = list(headersalign)
    tf_types = _table_format_types()
    is_tf = isinstance(tablefmt, tf_types)
    try:
        if not all(
            _simple_align(a) for a in (numalign, stralign, colglobalalign, headersglobalalign)
        ) or not all(
            _simple_align(a)
            for seq in (colalign, headersalign)
            if isinstance(seq, list)
            for a in seq
        ):
            raise _UnsupportedError("an alignment that is not a str")
        if type(MIN_PADDING) is not int:
            raise _UnsupportedError("a MIN_PADDING that is not an int")
        fmt_given = _describe(tablefmt) if is_tf else None
    except _UnsupportedError as e:
        real = _real_tabulate(str(e))
        return real.tabulate(tabular_data, **_for_real(real, options))  # type: ignore[no-any-return]

    if tabular_data is None:
        tabular_data = []

    list_of_lists, headers, headers_pad = _normalize_tabular_data(
        tabular_data, headers, showindex=showindex
    )
    normalized = (list_of_lists, headers[headers_pad:])
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


def _real_module() -> Any:
    real = sys.modules.get("tabulate")
    if getattr(real, "__pyroid__", False):
        real = getattr(real, "__wrapped__", None)
    return real


def _delegate_normalized(reason: str, normalized: Any, options: dict[str, Any]) -> str:
    """Hand a call whose input was already normalised to the real tabulate.

    The input may have been a generator, so tabulate gets the normalised rows and headers
    (which it normalises to themselves) and no index.
    """
    rows, headers = normalized
    real = _real_tabulate(reason)
    kwargs = dict(_for_real(real, options), headers=headers, showindex=False)
    return real.tabulate(rows, **kwargs)  # type: ignore[no-any-return]


def _for_real(real: Any, options: dict[str, Any]) -> dict[str, Any]:
    """`options` for the real tabulate: it only takes its own ``TableFormat`` class as one."""
    fmt = options["tablefmt"]
    real_cls = getattr(real, "TableFormat", None)
    if isinstance(fmt, TableFormat) and isinstance(real_cls, type) and real_cls is not TableFormat:
        return dict(options, tablefmt=real_cls(*fmt))
    return options


def _multiline_width(multiline_s: Any, line_width_fn: Any = len) -> Any:
    """Visible width of a potentially multiline content."""
    return max(map(line_width_fn, re.split("[\r\n]", multiline_s)))


def _align_column(
    strings: Any,
    alignment: Any,
    minwidth: Any = 0,
    has_invisible: Any = True,
    enable_widechars: Any = False,
    is_multiline: Any = False,
    preserve_whitespace: Any = False,
) -> list[str]:
    """``[string] -> [padded_string]``: tabulate's helper, run by the Rust aligner."""
    call = _Call(
        wcswidth=wcwidth.wcswidth if enable_widechars else None,  # type: ignore[union-attr]
        numpy_types=[],
        unicode_ok=_unicode_ok,
        max_digits=sys.get_int_max_str_digits(),
    )
    return _native._align_column(  # type: ignore[no-any-return]
        list(strings),
        alignment,
        minwidth,
        bool(has_invisible),
        bool(enable_widechars),
        bool(is_multiline),
        bool(preserve_whitespace),
        call,
    )


def _align_cell_veritically(
    text_lines: Any, num_lines: Any, column_width: Any, row_alignment: Any
) -> list[str]:
    """Pad a cell's lines to `num_lines` (tabulate's helper, run by the Rust code)."""
    return _native._align_cell_vertically(  # type: ignore[no-any-return]
        list(text_lines), num_lines, column_width, row_alignment
    )


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
    with sys.stdout if outfile == "-" else open(outfile, "w") as out:
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
#: The names of tabulate 0.10.0 the ``tabulate`` shim takes from here (the others forward to
#: the real tabulate when it is installed).
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
    "_ansi_codes",
    "_ansi_codes_bytes",
    "_ansi_color_reset_code",
    "_ansi_escape_pat",
    "_asciidoc_row",
    "_bool",
    "_build_simple_row",
    "_column_type",
    "_csi",
    "_esc",
    "_expand_iterable",
    "_expand_numparse",
    "_float_with_thousands_separators",
    "_format",
    "_grid_line_with_colons",
    "_grid_segment_with_colons",
    "_html_begin_table_without_header",
    "_html_row_with_attrs",
    "_is_file",
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
    "_multiline_width",
    "_normalize_tabular_data",
    "_osc",
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
    "_wrap_text_to_colwidths",
    "multiline_formats",
    "simple_separated_format",
    "tabulate",
    "tabulate_formats",
)

#: Module settings ``tabulate()`` reads on each call: writes on the shim land here too.
_SETTINGS = ("MIN_PADDING", "PRESERVE_WHITESPACE", "WIDE_CHARS_MODE", "wcwidth")

_shim_classes: dict[type, type] = {}


def _patched_names() -> dict[str, Any]:
    """The objects ``pyroid.patch("tabulate")`` puts on its ``tabulate`` shim."""
    here = globals()
    return {n: here[n] if n in here else getattr(_port, n) for n in _PATCHED_NAMES}


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
    _column_type,
    _isnumber,
    _more_generic,
    _port,
    _reinsert_separating_lines,
)
del _
