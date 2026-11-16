# Derived from CPython's Lib/difflib.py (PSF-2.0), (c) Python Software Foundation; modified.
"""A drop-in, Rust-backed replacement for the standard library's ``difflib``.

``import pyroid.difflib as difflib`` and use it as the standard library module. Results are the
running interpreter's own (CPython 3.11 to 3.15): the same ratios (the same float bits), matching
blocks, opcodes and diff lines. ``SequenceMatcher``, ``Differ`` and ``HtmlDiff`` subclass the
standard library's classes. See ``difflib.pyi``.

What runs natively: ``SequenceMatcher`` over a ``str``, ``bytes``, or a ``list``/``tuple`` of
``str``, ``int`` or ``bytes``; ``Differ.compare``, ``ndiff``, ``unified_diff``, ``context_diff``,
``get_close_matches`` and ``HtmlDiff``'s line pairing over lists or tuples of ``str``. Everything
else (other element types, a custom ``charjunk``, overridden methods) runs the standard library's
Python code, with this module's ``SequenceMatcher`` underneath where it applies.
"""

from __future__ import annotations

import contextlib
import difflib as _std
import sys
import types
import warnings
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from pyroid._pyroid import FallbackWarning
from pyroid._pyroid import difflib as _native

__all__ = [
    "IS_CHARACTER_JUNK",
    "IS_LINE_JUNK",
    "SUPPORTED_VERSIONS",
    "Differ",
    "HtmlDiff",
    "Match",
    "SequenceMatcher",
    "close_matches_batch",
    "context_diff",
    "diff_bytes",
    "get_close_matches",
    "ndiff",
    "ratio_batch",
    "restore",
    "unified_diff",
]

#: The CPython versions whose difflib this module reproduces.
SUPPORTED_VERSIONS = ((3, 11), (3, 12), (3, 13), (3, 14), (3, 15))

_KNOWN = sys.version_info[:2] in SUPPORTED_VERSIONS
# Differ._fancy_replace: 3.14 searches a window around the diagonal (gh-119105).
_WINDOWED = sys.version_info >= (3, 14)

# Everything the standard library module defines (private helpers included, so that its
# functions rebound below find them here), then this module's own versions on top.
globals().update(
    {k: v for k, v in vars(_std).items() if not (k.startswith("__") and k.endswith("__"))}
)

IS_CHARACTER_JUNK = _std.IS_CHARACTER_JUNK
IS_LINE_JUNK = _std.IS_LINE_JUNK
Match = _std.Match
restore = _std.restore
diff_bytes = _std.diff_bytes
_check_types = _std._check_types

_WARNED: set[str] = set()


def _fallback(name: str, why: str) -> None:
    """The once-per-process FallbackWarning for `name`."""
    if name in _WARNED:
        return
    _WARNED.add(name)
    # stacklevel 2 is our caller; point at the first frame outside this module and difflib.
    frame = sys._getframe(1)
    level = 2
    while frame.f_back is not None and frame.f_globals.get("__name__") in (__name__, "difflib"):
        frame = frame.f_back
        level += 1
    warnings.warn(
        f"pyroid.difflib.{name}: {why} run the standard library's difflib code (slow path). "
        "This warning is shown once.",
        FallbackWarning,
        stacklevel=level,
    )


def _rebind(f: Any) -> Any:
    """`f` (a function of the standard library's difflib) resolving its globals here."""
    g = types.FunctionType(f.__code__, globals(), f.__name__, f.__defaults__, f.__closure__)
    g.__kwdefaults__ = f.__kwdefaults__
    g.__qualname__ = f.__qualname__
    g.__doc__ = f.__doc__
    g.__module__ = __name__
    g.__dict__.update(f.__dict__)
    return g


# ---------------------------------------------------------------------------
# SequenceMatcher
# ---------------------------------------------------------------------------

_StdSequenceMatcher = _std.SequenceMatcher
_SM = {k: _StdSequenceMatcher.__dict__[k] for k in _StdSequenceMatcher.__dict__}
_chain_b = _SM["_SequenceMatcher__chain_b"]
_ELEMENTS = "these sequences (elements other than str, int and bytes, or other sequence types)"


def _init(
    self: Any,
    isjunk: Callable[[Any], bool] | None = None,
    a: Any = "",
    b: Any = "",
    autojunk: bool = True,
) -> None:
    cls = type(self)
    if cls.set_seqs in _SET_SEQS and cls.set_seq1 is _set_seq1 and cls.set_seq2 is _set_seq2:
        if not _native._sm_init(self, isjunk, a, b, autojunk, cls in _PLAIN):
            self.matching_blocks = None
            self.fullbcount = None
            _fallback("SequenceMatcher", _ELEMENTS)
            _chain_b(self)
    else:
        _SM["__init__"](self, isjunk, a, b, autojunk)


def _set_seq1(self: Any, a: Any) -> None:
    if a is self.a:
        return
    self.a = a
    self.opcodes = None
    d = self.__dict__
    d.pop("matching_blocks", None)
    d.pop("_pyroid_mb", None)


def _set_seq2(self: Any, b: Any) -> None:
    if b is self.b:
        return
    if not _native._sm_set_seq2(self, b):
        self.matching_blocks = None
        self.fullbcount = None
        _fallback("SequenceMatcher", _ELEMENTS)
        _chain_b(self)


def _find_longest_match(
    self: Any, alo: int = 0, ahi: int | None = None, blo: int = 0, bhi: int | None = None
) -> Any:
    m = _native._sm_find_longest_match(self, alo, ahi, blo, bhi)
    if m is None:
        return _SM["find_longest_match"](self, alo, ahi, blo, bhi)
    return m


def _slow(self: Any) -> None:
    # set_seq2 warned if b's elements are the reason; here it is a.
    if "_pyroid_b" in self.__dict__:
        _fallback("SequenceMatcher", _ELEMENTS)


def _get_matching_blocks(self: Any) -> Any:
    blocks = self.matching_blocks
    if blocks is not None:
        return blocks
    if type(self).find_longest_match is _find_longest_match:
        found = _native._sm_matching_blocks(self)
        if found is not None:
            return found
        _slow(self)
    return _SM["get_matching_blocks"](self)


def _native_blocks(self: Any) -> bool:
    """True if the blocks can be computed natively.

    That is: neither get_matching_blocks nor find_longest_match is overridden, and no
    matching_blocks list is set.
    """
    cls = type(self)
    return (
        cls.get_matching_blocks is _get_matching_blocks
        and cls.find_longest_match is _find_longest_match
        and self.__dict__.get("matching_blocks") is None
    )


def _get_opcodes(self: Any) -> Any:
    if self.opcodes is not None:
        return self.opcodes
    if _native_blocks(self):
        found = _native._sm_opcodes(self)
        if found is not None:
            return found
        _slow(self)
    return _SM["get_opcodes"](self)


def _ratio(self: Any) -> float:
    if _native_blocks(self):
        r = _native._sm_ratio(self)
        if r is not None:
            return r  # type: ignore[no-any-return]
        _slow(self)
    return _SM["ratio"](self)  # type: ignore[no-any-return]


def _quick_ratio(self: Any) -> float:
    if self.__dict__.get("fullbcount") is None:
        r = _native._sm_quick_ratio(self)
        if r is not None:
            return r  # type: ignore[no-any-return]
        _slow(self)
    return _SM["quick_ratio"](self)  # type: ignore[no-any-return]


def _getstate(self: Any) -> dict[str, Any]:
    # CPython's state (the same keys, so plain difflib can load it): the lazy attributes
    # made, the native index left out.
    state = {k: v for k, v in self.__dict__.items() if not k.startswith("_pyroid_")}
    for name in _LAZY:
        if name not in state:
            with contextlib.suppress(AttributeError):
                state[name] = getattr(self, name)
    return state


class _Lazy:
    """A lazy attribute of a matcher whose index is native.

    ``b2j``, ``bjunk``, ``bpopular``, ``fullbcount`` or ``matching_blocks``: made (in CPython's
    order) on first read, then a plain instance attribute.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __get__(self, obj: Any, objtype: Any = None) -> Any:
        if obj is None:
            return self
        name = self._name
        if name == "matching_blocks":
            return _native._sm_cached_blocks(obj)
        value = _native._sm_chain_attr(obj, name)
        if value is None and name != "fullbcount":
            raise AttributeError(f"{type(obj).__name__!r} object has no attribute {name!r}")
        return value


_LAZY = ("b2j", "bjunk", "bpopular", "fullbcount", "matching_blocks")

#: What pyroid sets on a SequenceMatcher class (its own, or difflib's under pyroid.patch).
_SM_METHODS: dict[str, Any] = {
    "__init__": _init,
    "set_seq1": _set_seq1,
    "set_seq2": _set_seq2,
    "find_longest_match": _find_longest_match,
    "get_matching_blocks": _get_matching_blocks,
    "get_opcodes": _get_opcodes,
    "ratio": _ratio,
    "quick_ratio": _quick_ratio,
    "__getstate__": _getstate,
    **{name: _Lazy(name) for name in _LAZY},
}

for _name, _f in _SM_METHODS.items():
    if isinstance(_f, types.FunctionType):
        _f.__name__ = _name
        _f.__qualname__ = f"SequenceMatcher.{_name}"
        _f.__doc__ = getattr(_StdSequenceMatcher, _name).__doc__


class SequenceMatcher(_StdSequenceMatcher):
    __doc__ = _StdSequenceMatcher.__doc__


for _name, _f in _StdSequenceMatcher.__dict__.items():
    # The methods pyroid does not replace, rebound (same code), so their doctests run here too.
    if isinstance(_f, types.FunctionType) and not _name.startswith("_SequenceMatcher"):
        setattr(SequenceMatcher, _name, _rebind(_f))
for _name, _f in _SM_METHODS.items():
    setattr(SequenceMatcher, _name, _f)
# set_seqs as difflib.SequenceMatcher has it (under pyroid.patch) and as rebound here.
_SET_SEQS = (_SM["set_seqs"], SequenceMatcher.__dict__["set_seqs"])
# Classes whose attributes are plain instance-dict entries (no subclass __setattr__ or properties).
_PLAIN = (SequenceMatcher, _StdSequenceMatcher)

# ---------------------------------------------------------------------------
# Differ, ndiff
# ---------------------------------------------------------------------------

_HOOKS = ("_dump", "_plain_replace", "_fancy_replace", "_fancy_helper", "_qformat")


def _junk_kinds(linejunk: Any, charjunk: Any) -> tuple[int, bool] | None:
    """(linejunk kind, charjunk is IS_CHARACTER_JUNK) for the native paths, or None."""
    if charjunk is not None and charjunk is not IS_CHARACTER_JUNK:
        return None
    kind = 0 if linejunk is None else 1 if linejunk is IS_LINE_JUNK else 2
    return kind, charjunk is not None


def _make_compare(fallback: Any, hooks: dict[str, Any]) -> Any:
    def compare(self: Any, a: Sequence[str], b: Sequence[str]) -> Iterator[str]:
        cls = type(self)
        kinds = _junk_kinds(self.linejunk, self.charjunk)
        if kinds is not None and all(getattr(cls, h) is f for h, f in hooks.items()):
            lines = _native._differ_compare(a, b, kinds[0], self.linejunk, kinds[1], _WINDOWED)
            if lines is not None:
                yield from lines
                return
        yield from fallback(self, a, b)

    compare.__doc__ = fallback.__doc__
    compare.__qualname__ = "Differ.compare"
    return compare


class Differ(_std.Differ):
    __doc__ = _std.Differ.__doc__


for _name, _f in _std.Differ.__dict__.items():
    if isinstance(_f, types.FunctionType):
        setattr(Differ, _name, _rebind(_f))
Differ.compare = _make_compare(  # type: ignore[method-assign]
    Differ.__dict__["compare"], {h: Differ.__dict__[h] for h in _HOOKS}
)

# What pyroid.patch("difflib") sets as difflib.Differ.compare: the same fast path, with
# difflib's own methods as the hooks and the fallback.
_PATCH_COMPARE = _make_compare(
    _std.Differ.__dict__["compare"], {h: _std.Differ.__dict__[h] for h in _HOOKS}
)

ndiff = _rebind(_std.ndiff)

# ---------------------------------------------------------------------------
# unified_diff, context_diff
# ---------------------------------------------------------------------------

_std_unified = _rebind(_std.unified_diff)
_std_context = _rebind(_std.context_diff)
_COLOR = "color" in (_std.unified_diff.__kwdefaults__ or {})


def _theme(color: bool) -> tuple[str, ...]:
    # As 3.15's unified_diff picks it.
    if color and _std.can_colorize():
        t = _std.get_theme(force_color=True).difflib
    else:
        t = _std.get_theme(force_no_color=True).difflib
    return (t.header, t.hunk, t.context, t.removed, t.added, t.reset)


if _COLOR:

    def unified_diff(
        a: Sequence[str],
        b: Sequence[str],
        fromfile: str = "",
        tofile: str = "",
        fromfiledate: str = "",
        tofiledate: str = "",
        n: int = 3,
        lineterm: str = "\n",
        *,
        color: bool = False,
    ) -> Iterator[str]:
        _check_types(a, b, fromfile, tofile, fromfiledate, tofiledate, lineterm)
        theme = _theme(color)
        lines = _native._line_diff(
            a, b, fromfile, tofile, fromfiledate, tofiledate, n, lineterm, False, theme
        )
        if lines is None:
            yield from _std_unified(
                a, b, fromfile, tofile, fromfiledate, tofiledate, n, lineterm, color=color
            )
            return
        yield from lines

else:

    def unified_diff(  # type: ignore[misc]
        a: Sequence[str],
        b: Sequence[str],
        fromfile: str = "",
        tofile: str = "",
        fromfiledate: str = "",
        tofiledate: str = "",
        n: int = 3,
        lineterm: str = "\n",
    ) -> Iterator[str]:
        _check_types(a, b, fromfile, tofile, fromfiledate, tofiledate, lineterm)
        lines = _native._line_diff(
            a, b, fromfile, tofile, fromfiledate, tofiledate, n, lineterm, False
        )
        if lines is None:
            yield from _std_unified(a, b, fromfile, tofile, fromfiledate, tofiledate, n, lineterm)
            return
        yield from lines


def context_diff(
    a: Sequence[str],
    b: Sequence[str],
    fromfile: str = "",
    tofile: str = "",
    fromfiledate: str = "",
    tofiledate: str = "",
    n: int = 3,
    lineterm: str = "\n",
) -> Iterator[str]:
    _check_types(a, b, fromfile, tofile, fromfiledate, tofiledate, lineterm)
    lines = _native._line_diff(a, b, fromfile, tofile, fromfiledate, tofiledate, n, lineterm, True)
    if lines is None:
        yield from _std_context(a, b, fromfile, tofile, fromfiledate, tofiledate, n, lineterm)
        return
    yield from lines


unified_diff.__doc__ = _std.unified_diff.__doc__
context_diff.__doc__ = _std.context_diff.__doc__

# ---------------------------------------------------------------------------
# get_close_matches
# ---------------------------------------------------------------------------

_std_close = _rebind(_std.get_close_matches)


def get_close_matches(
    word: Sequence[Any],
    possibilities: Any,
    n: int = 3,
    cutoff: float = 0.6,
    *,
    threads: int | None = None,
) -> list[Any]:
    if not n > 0:
        raise ValueError(f"n must be > 0: {n!r}")
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError(f"cutoff must be in [0.0, 1.0]: {cutoff!r}")
    if type(possibilities) is not list and type(possibilities) is not tuple:
        possibilities = list(possibilities)
    found = _native._get_close_matches(word, possibilities, n, cutoff, threads)
    if found is None:
        return _std_close(word, possibilities, n, cutoff)  # type: ignore[no-any-return]
    return found  # type: ignore[no-any-return]


get_close_matches.__doc__ = _std.get_close_matches.__doc__

# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------

ratio_batch = _native.ratio_batch


def close_matches_batch(
    words: Any, possibilities: Any, n: int = 3, cutoff: float = 0.6, *, threads: int | None = None
) -> list[list[str] | None]:
    if not n > 0:
        raise ValueError(f"n must be > 0: {n!r}")
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError(f"cutoff must be in [0.0, 1.0]: {cutoff!r}")
    return _native._close_matches_batch(  # type: ignore[no-any-return]
        words, possibilities, min(int(n), sys.maxsize), float(cutoff), threads
    )


# ---------------------------------------------------------------------------
# HtmlDiff and _mdiff
# ---------------------------------------------------------------------------

_std_mdiff = _rebind(_std._mdiff)


def _mdiff(
    fromlines: Sequence[str],
    tolines: Sequence[str],
    context: int | None = None,
    linejunk: Callable[[str], bool] | None = None,
    charjunk: Callable[[str], bool] | None = IS_CHARACTER_JUNK,
) -> Iterator[Any]:
    kinds = _junk_kinds(linejunk, charjunk)
    if kinds is not None and (context is None or (type(context) is int and context >= 0)):
        rows = _native._mdiff(fromlines, tolines, context, kinds[0], linejunk, kinds[1], _WINDOWED)
        if rows is not None:
            yield from rows
            return
    yield from _std_mdiff(fromlines, tolines, context, linejunk, charjunk)


_mdiff.__doc__ = _std._mdiff.__doc__


_HTML_HOOKS = (
    "_make_prefix",
    "_tab_newline_replace",
    "_line_wrapper",
    "_split_line",
    "_collect_lines",
    "_format_line",
    "_convert_flags",
)
# 3.15's _format_line marks changed lines with class="diff_changed".
_CHANGED_CLASS = sys.version_info >= (3, 15)


def _html_marks(text: str) -> str:
    """``make_table``'s final replacements of the change marks."""
    return (
        text.replace("\0+", '<span class="diff_add">')
        .replace("\0-", '<span class="diff_sub">')
        .replace("\0^", '<span class="diff_chg">')
        .replace("\1", "</span>")
        .replace("\t", "&nbsp;")
    )


def _make_make_table(fallback: Any, hooks: dict[str, Any]) -> Any:
    def make_table(
        self: Any,
        fromlines: Sequence[str],
        tolines: Sequence[str],
        fromdesc: str = "",
        todesc: str = "",
        context: bool = False,
        numlines: int = 5,
    ) -> str:
        cls = type(self)
        kinds = _junk_kinds(self._linejunk, self._charjunk)
        tabsize, wrap, template = self._tabsize, self._wrapcolumn, self._table_template
        rows = None
        if (
            kinds is not None
            and type(template) is str
            and template.count("%(data_rows)s") == 1
            and "%%" not in template
            and type(tabsize) is int
            and (wrap is None or (type(wrap) is int and wrap >= 0))
            and type(numlines) is int
            and numlines >= 0
            and all(getattr(cls, h) is f for h, f in hooks.items())
        ):
            with_context = bool(context)
            try:
                rows = _native._html_prepare(
                    fromlines,
                    tolines,
                    tabsize,
                    wrap or 0,
                    numlines if with_context else None,
                    kinds[0],
                    self._linejunk,
                    kinds[1],
                    _WINDOWED,
                )
            except BaseException:
                self._make_prefix()  # CPython had made the prefixes before it failed
                raise
        if rows is None:
            return fallback(self, fromlines, tolines, fromdesc, todesc, context, numlines)  # type: ignore[no-any-return]
        # The rest of CPython's make_table, around the native rows.
        self._make_prefix()
        data_rows = rows.render(
            self._prefix[0], self._prefix[1], with_context, numlines, _CHANGED_CLASS
        )
        if fromdesc or todesc:
            header_row = "<thead><tr>%s%s%s%s</tr></thead>" % (  # noqa: UP031 - CPython's formatting
                '<th class="diff_next"><br /></th>',
                '<th colspan="2" class="diff_header">%s</th>' % fromdesc,  # noqa: UP031
                '<th class="diff_next"><br /></th>',
                '<th colspan="2" class="diff_header">%s</th>' % todesc,  # noqa: UP031
            )
        else:
            header_row = ""
        # The template around the rows gets the final replacements here; the rows have had
        # them (no replacement can span a row's ends).
        head, _, tail = template.partition("%(data_rows)s")
        parts = {"header_row": header_row, "prefix": self._prefix[1]}
        return _html_marks(head % parts) + data_rows + _html_marks(tail % parts)  # type: ignore[no-any-return]

    make_table.__doc__ = fallback.__doc__
    make_table.__qualname__ = "HtmlDiff.make_table"
    return make_table


class HtmlDiff(_std.HtmlDiff):
    __doc__ = _std.HtmlDiff.__doc__


for _name, _f in _std.HtmlDiff.__dict__.items():
    if isinstance(_f, types.FunctionType):
        setattr(HtmlDiff, _name, _rebind(_f))
HtmlDiff.make_table = _make_make_table(  # type: ignore[method-assign]
    HtmlDiff.__dict__["make_table"], {h: HtmlDiff.__dict__[h] for h in _HTML_HOOKS}
)
# What pyroid.patch("difflib") sets as difflib.HtmlDiff.make_table.
_PATCH_MAKE_TABLE = _make_make_table(
    _std.HtmlDiff.__dict__["make_table"], {h: _std.HtmlDiff.__dict__[h] for h in _HTML_HOOKS}
)

for _cls in (SequenceMatcher, Differ, HtmlDiff):
    _cls.__module__ = __name__

del _name, _f, _cls

#: What pyroid.patch("difflib") sets on the standard library's module.
_PATCH_FUNCTIONS = ("get_close_matches", "unified_diff", "context_diff", "_mdiff")

# The fallbacks' docstrings are the public functions' (their doctests run there once).
for _f in (_std_unified, _std_context, _std_close, _std_mdiff, _PATCH_COMPARE, _PATCH_MAKE_TABLE):
    _f.__doc__ = None
del _f

if not _KNOWN:  # an interpreter whose difflib this module does not know: the standard library's
    SequenceMatcher = _std.SequenceMatcher  # type: ignore[misc]
    Differ = _std.Differ  # type: ignore[misc]
    HtmlDiff = _std.HtmlDiff  # type: ignore[misc]
    ndiff = _std.ndiff
    unified_diff = _std.unified_diff
    context_diff = _std.context_diff
    get_close_matches = _std.get_close_matches  # type: ignore[assignment]
    _mdiff = _std._mdiff
    warnings.warn(
        f"pyroid.difflib reproduces difflib of CPython 3.11 to 3.15, not "
        f"{sys.version_info[0]}.{sys.version_info[1]}'s: it is the standard library's difflib "
        "here (slow path).",
        FallbackWarning,
        stacklevel=2,
    )
