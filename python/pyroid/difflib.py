# Derived from CPython's Lib/difflib.py (PSF-2.0), (c) Python Software Foundation; modified.
"""A drop-in, Rust-backed replacement for the standard library's ``difflib``.

``import pyroid.difflib as difflib`` and use it as the standard library module. Results are the
running interpreter's own (CPython 3.11 to 3.15): the same ratios (the same float bits), matching
blocks, opcodes and diff lines. ``SequenceMatcher``, ``Differ`` and ``HtmlDiff`` subclass the
standard library's classes. See ``difflib.pyi``.

What runs natively: ``SequenceMatcher`` over a ``str``, ``bytes``, ``bytearray``, or a
``list``/``tuple`` of ``str``, ``int`` (any size), ``bool`` or ``bytes`` (subclasses too, if they
keep their comparisons and indexing); ``get_close_matches`` over ``str``; ``Differ.compare``,
``ndiff``, ``unified_diff``, ``context_diff`` and ``HtmlDiff``'s line pairing over lists or tuples
of ``str``, or a ``str`` (one line per character). Everything else (other element types, ``str``
mixed with ``bytes`` under ``python -b``, a custom ``charjunk``, overridden methods, one of this
module's classes or functions replaced by another) runs the standard library's Python code, with
this module's ``SequenceMatcher`` underneath where it applies.
"""

from __future__ import annotations

import contextlib
import difflib as _std
import operator
import os
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
_UNSET = object()


def _fallback_sm(self: Any) -> None:
    """The once-per-process FallbackWarning for SequenceMatcher, saying what triggered it."""
    if "SequenceMatcher" in _WARNED:
        return
    _WARNED.add("SequenceMatcher")
    # stacklevel 2 is our caller; point at the first frame outside this module and difflib.
    frame = sys._getframe(1)
    level = 2
    while frame.f_back is not None and frame.f_globals.get("__name__") in (__name__, "difflib"):
        frame = frame.f_back
        level += 1
    warnings.warn(
        f"pyroid.difflib.SequenceMatcher: {_native._sm_why(self)} runs the standard library's "
        "difflib code (slow path). A str, bytes or bytearray, or a list or tuple of str, int "
        "(bool too) and bytes, runs natively, and so do subclasses of these that keep their "
        "comparisons and indexing. This warning is shown once.",
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


def _init(
    self: Any,
    isjunk: Callable[[Any], bool] | None = None,
    a: Any = "",
    b: Any = "",
    autojunk: bool = True,
) -> None:
    cls = type(self)
    if (
        cls.set_seqs in _SET_SEQS
        and cls.set_seq1 is _set_seq1
        and cls.set_seq2 is _set_seq2
        and cls._SequenceMatcher__chain_b is _chain_b
    ):
        done = _native._sm_init(self, isjunk, a, b, autojunk, cls in _PLAIN)
        if done:
            return
        if done is None:  # difflib's class, and the patch has just been switched off
            _SM["__init__"](self, isjunk, a, b, autojunk)
            return
        self.matching_blocks = None
        self.fullbcount = None
        _fallback_sm(self)
        _chain_b(self)
    else:
        _SM["__init__"](self, isjunk, a, b, autojunk)


def _set_seq1(self: Any, a: Any) -> None:
    if _native._sm_set_seq1(self, a) is None:
        _SM["set_seq1"](self, a)


def _set_seq2(self: Any, b: Any) -> None:
    if type(self)._SequenceMatcher__chain_b is not _chain_b:
        # A subclass post-processes b2j: CPython's set_seq2 calls its hook.
        _SM["set_seq2"](self, b)
        return
    if b is self.b:
        return
    done = _native._sm_set_seq2(self, b)
    if done:
        return
    if done is None:
        _SM["set_seq2"](self, b)
        return
    self.matching_blocks = None
    self.fullbcount = None
    _fallback_sm(self)
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
        _fallback_sm(self)


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
            value = _native._sm_cached_blocks(obj)
        else:
            value = _native._sm_chain_attr(obj, name)
        if value is None:
            # pyroid.unpatch("difflib") may have just made it a plain attribute.
            plain = obj.__dict__.get(name, _UNSET)
            if plain is not _UNSET:
                return plain
            if name not in ("matching_blocks", "fullbcount"):
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


# For pyroid.patch("difflib") and pyroid.unpatch("difflib") (_build_difflib in
# pyroid/integrations/_patch.py); both are native (crates/pyroid-py/src/difflib/settle.rs).
#
# _switch_on(swaps, missing) sets each (owner, name, value) of `swaps` (or none, if one cannot
# be set), keeps what it replaced, and only then lets pyroid's methods give difflib's matchers
# native state. False if its attributes are still in place from an earlier call.
#
# _switch_off() undoes it. A matcher of difflib's class made or used under the patch keeps
# b2j, bjunk, bpopular, fullbcount and matching_blocks in pyroid's native state, which only
# pyroid's descriptors on the class can read. Each one becomes CPython's state (plain instance
# attributes, as pickling gives them), now or, if another thread's __init__, set_seq1 or
# set_seq2 is changing it, when that call ends; and a call that starts from now on runs
# CPython's method. Then every replaced attribute goes back, all of them even if a matcher
# could not be settled (that error is raised afterwards). Returns how many matchers were
# settled now (0, doing nothing, if the patch is not in place). The matchers are recorded as
# they get native state, so this does not walk the heap or wait for other threads.
#
# No Python code runs between their steps, so Ctrl-C cannot stop either half-way.
_switch_on: Callable[[tuple[tuple[Any, str, Any], ...], object], bool] = _native._sm_patch
_switch_off: Callable[[], int] = _native._sm_unpatch


class SequenceMatcher(_StdSequenceMatcher):
    __doc__ = _StdSequenceMatcher.__doc__


_native._sm_own(SequenceMatcher)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_native._sm_forked)


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


def _make_compare(fallback: Any, hooks: dict[str, Any], g: dict[str, Any], sm: Any) -> Any:
    """`Differ.compare` for the module whose globals are `g`.

    CPython's `compare` looks up `SequenceMatcher` there, which must be `sm` for the native path.
    """

    def compare(self: Any, a: Sequence[str], b: Sequence[str]) -> Iterator[str]:
        cls = type(self)
        kinds = _junk_kinds(self.linejunk, self.charjunk)
        try:
            native = (
                kinds is not None
                and g["SequenceMatcher"] is sm
                and all(getattr(cls, h) is f for h, f in hooks.items())
            )
        except KeyError:  # deleted: CPython's code raises NameError
            native = False
        if native and kinds is not None:
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
    Differ.__dict__["compare"],
    {h: Differ.__dict__[h] for h in _HOOKS},
    globals(),
    SequenceMatcher,
)

# What pyroid.patch("difflib") sets as difflib.Differ.compare: the same fast path, with
# difflib's own methods as the hooks and the fallback, and difflib's globals.
_PATCH_COMPARE = _make_compare(
    _std.Differ.__dict__["compare"],
    {h: _std.Differ.__dict__[h] for h in _HOOKS},
    _std.__dict__,
    _std.SequenceMatcher,
)

ndiff = _rebind(_std.ndiff)


class _Module:
    """The globals the native paths of one module's functions depend on.

    Each native path stands in for the standard library's Python code, which looks up some of
    its module's globals when it runs: `SequenceMatcher` (which cdifflib tells its users to
    replace), `Differ`, `ndiff`, `_mdiff`. A native path runs only while they are the module's
    own: pyroid.difflib's for its functions; difflib's for those pyroid.patch("difflib") sets
    there. Otherwise the standard library's code runs with the replacement.
    """

    __slots__ = ("differ", "differ_funcs", "g", "get_differ_funcs", "mdiff", "ndiff", "sm")

    def __init__(
        self, g: dict[str, Any], sm: Any, differ: Any, ndiff: Any, compare: Any = None
    ) -> None:
        self.g = g
        self.sm = sm
        self.differ = differ
        self.ndiff = ndiff
        # What ndiff's Differ(linejunk, charjunk).compare(a, b) runs (`compare`: what the patch
        # sets there), read in one call.
        names = ("__init__", "compare", *_HOOKS)
        self.get_differ_funcs = operator.attrgetter(*names)
        funcs = [differ.__dict__[n] for n in names]
        if compare is not None:
            funcs[1] = compare
        self.differ_funcs = tuple(funcs)
        self.mdiff: Any = None  # set once that module's _mdiff exists

    def ndiff_ok(self) -> bool:
        """Whether `ndiff` (with `Differ` and `SequenceMatcher` under it) is the module's own."""
        g = self.g
        try:
            if (
                g["ndiff"] is not self.ndiff
                or g["Differ"] is not self.differ
                or g["SequenceMatcher"] is not self.sm
            ):
                return False
        except KeyError:  # deleted: CPython's code raises NameError
            return False
        return all(map(operator.is_, self.get_differ_funcs(self.differ), self.differ_funcs))

    def mdiff_ok(self) -> bool:
        """Whether `_mdiff`, and what it runs, is the module's own."""
        try:
            if self.g["_mdiff"] is not self.mdiff:
                return False
        except KeyError:
            return False
        return self.ndiff_ok()


def _placed(f: Any, m: _Module) -> Any:
    """`f`, named as the module-level function of `m` it is, so that pickle finds it there.

    That is pyroid.difflib, or (patched) difflib, which unpatched processes load as difflib's own.
    """
    f.__qualname__ = f.__name__
    f.__module__ = m.g["__name__"]
    return f


_OWN = _Module(globals(), SequenceMatcher, Differ, ndiff)
# Under the patch, difflib's Differ.compare is pyroid's (see _build_difflib).
_STD = _Module(_std.__dict__, _std.SequenceMatcher, _std.Differ, _std.ndiff, _PATCH_COMPARE)
# The standard library's own functions, for the patched ones to fall back to (they look up
# difflib's globals, as difflib's functions do).
_STD_FUNCTIONS = {
    n: getattr(_std, n) for n in ("get_close_matches", "unified_diff", "context_diff", "_mdiff")
}

# ---------------------------------------------------------------------------
# unified_diff, context_diff
# ---------------------------------------------------------------------------

_std_unified = _rebind(_std.unified_diff)
_std_context = _rebind(_std.context_diff)
_COLOR = "color" in (_std.unified_diff.__kwdefaults__ or {})
_UNIFIED_DOC = _std.unified_diff.__doc__
_CONTEXT_DOC = _std.context_diff.__doc__


def _theme(color: bool) -> tuple[str, ...]:
    # As 3.15's unified_diff picks it.
    if color and _std.can_colorize():
        t = _std.get_theme(force_color=True).difflib
    else:
        t = _std.get_theme(force_no_color=True).difflib
    return (t.header, t.hunk, t.context, t.removed, t.added, t.reset)


def _make_unified(m: _Module, fallback: Any) -> Any:
    """`unified_diff` for module `m`, running `fallback` where the native path does not."""
    g, sm = m.g, m.sm
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
            g["_check_types"](a, b, fromfile, tofile, fromfiledate, tofiledate, lineterm)
            lines = None
            try:
                native = g["SequenceMatcher"] is sm
            except KeyError:  # deleted: CPython's code raises NameError
                native = False
            if native:
                theme = _theme(color)
                lines = _native._line_diff(
                    a, b, fromfile, tofile, fromfiledate, tofiledate, n, lineterm, False, theme
                )
            if lines is None:
                yield from fallback(
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
            g["_check_types"](a, b, fromfile, tofile, fromfiledate, tofiledate, lineterm)
            lines = None
            try:
                native = g["SequenceMatcher"] is sm
            except KeyError:  # deleted: CPython's code raises NameError
                native = False
            if native:
                lines = _native._line_diff(
                    a, b, fromfile, tofile, fromfiledate, tofiledate, n, lineterm, False
                )
            if lines is None:
                yield from fallback(a, b, fromfile, tofile, fromfiledate, tofiledate, n, lineterm)
                return
            yield from lines

    unified_diff.__doc__ = _UNIFIED_DOC
    return _placed(unified_diff, m)


def _make_context(m: _Module, fallback: Any) -> Any:
    """`context_diff` for module `m`, running `fallback` where the native path does not."""
    g, sm = m.g, m.sm

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
        g["_check_types"](a, b, fromfile, tofile, fromfiledate, tofiledate, lineterm)
        lines = None
        try:
            native = g["SequenceMatcher"] is sm
        except KeyError:  # deleted: CPython's code raises NameError
            native = False
        if native:
            lines = _native._line_diff(
                a, b, fromfile, tofile, fromfiledate, tofiledate, n, lineterm, True
            )
        if lines is None:
            yield from fallback(a, b, fromfile, tofile, fromfiledate, tofiledate, n, lineterm)
            return
        yield from lines

    context_diff.__doc__ = _CONTEXT_DOC
    return _placed(context_diff, m)


unified_diff = _make_unified(_OWN, _std_unified)
context_diff = _make_context(_OWN, _std_context)

# ---------------------------------------------------------------------------
# get_close_matches
# ---------------------------------------------------------------------------

_std_close = _rebind(_std.get_close_matches)


def _make_close(m: _Module, fallback: Any) -> Any:
    """`get_close_matches` for module `m`, running `fallback` where the native path does not."""
    g, sm = m.g, m.sm

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
        found = None
        try:
            native = g["SequenceMatcher"] is sm
        except KeyError:  # deleted: CPython's code raises NameError
            native = False
        if native:
            found = _native._get_close_matches(word, possibilities, n, cutoff, threads)
        if found is None:
            return fallback(word, possibilities, n, cutoff)  # type: ignore[no-any-return]
        return found  # type: ignore[no-any-return]

    get_close_matches.__doc__ = _std.get_close_matches.__doc__
    return _placed(get_close_matches, m)


get_close_matches = _make_close(_OWN, _std_close)

# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------


def ratio_batch(
    a: Any,
    b: Any,
    *,
    isjunk: Callable[[str], object] | None = None,
    autojunk: bool = True,
    threads: int | None = None,
) -> list[float | None]:
    """`SequenceMatcher(isjunk, a[i], b[i], autojunk).ratio()` for every `i`, on all cores.

    ```python
    ratio_batch(["Acme Corp.", "Globex"], ["ACME Corporation", "Globex Inc"])
    ratio_batch("appel", candidates)  # one str is compared with every element of the other
    ```

    Args:
        a: The first sequences: a `str`, a list or tuple of `str`, or an Arrow string array.
        b: The second sequences, as `a`; both must have the same length unless one is a `str`.
        isjunk: `None`, or a function of one character, called once per distinct character of
            `b` before the work starts (so it must not depend on anything else).
        autojunk: As for `SequenceMatcher`.
        threads: `None`: the whole shared pool; `n`: at most `n` tasks at once; `1`: this thread.

    Returns:
        A list of floats, the same as `SequenceMatcher`'s; `None` where an Arrow input is null.

    Raises:
        ValueError: `a` and `b` have different lengths.
        pyroid.InputError: An argument is not text (lone surrogates raise `UnicodeEncodeError`).
    """
    return _native.ratio_batch(  # type: ignore[no-any-return]
        a, b, isjunk=isjunk, autojunk=autojunk, threads=threads
    )


def close_matches_batch(
    words: Any, possibilities: Any, n: int = 3, cutoff: float = 0.6, *, threads: int | None = None
) -> list[list[str] | None]:
    """`get_close_matches(word, possibilities, n, cutoff)` for every word, on all cores.

    Args:
        words: The words to look up: a `str`, a list or tuple of `str`, or an Arrow string array.
        possibilities: The candidates: a list or tuple of `str`, or an Arrow string array (nulls
            are skipped).
        n: At most this many results per word. With lists and tuples, a non-integer `n` gives
            what the loop above gives (its `heapq.nlargest` decides, by Python version).
        cutoff: The lowest ratio kept, in `[0.0, 1.0]`.
        threads: `None`: the whole shared pool; `n`: at most `n` tasks at once; `1`: this thread.

    Returns:
        One list per word (the possibilities' own `str` objects for a list or tuple), or `None`
        for a null word.

    Raises:
        ValueError: `n <= 0` or `cutoff` outside `[0.0, 1.0]`.
        TypeError: `n` is not an integer and an input is an Arrow array.
        pyroid.InputError: An argument is not text.
    """
    if not n > 0:
        raise ValueError(f"n must be > 0: {n!r}")
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError(f"cutoff must be in [0.0, 1.0]: {cutoff!r}")
    try:
        n = operator.index(n)
    except TypeError:
        # heapq.nlargest decides what a non-integer n does, and that differs between versions
        # (3.15 takes 3.0, or 2.5 when there are no more matches): the loop itself answers.
        if type(possibilities) in (list, tuple) and type(words) in (str, list, tuple):
            words = [words] if type(words) is str else words
            return [get_close_matches(w, possibilities, n, cutoff) for w in words]
        raise
    return _native._close_matches_batch(  # type: ignore[no-any-return]
        words, possibilities, min(n, sys.maxsize), float(cutoff), threads
    )


# ---------------------------------------------------------------------------
# HtmlDiff and _mdiff
# ---------------------------------------------------------------------------

_std_mdiff = _rebind(_std._mdiff)


def _make_mdiff(m: _Module, fallback: Any) -> Any:
    """`_mdiff` for module `m`, running `fallback` where the native path does not."""

    def _mdiff(
        fromlines: Sequence[str],
        tolines: Sequence[str],
        context: int | None = None,
        linejunk: Callable[[str], bool] | None = None,
        charjunk: Callable[[str], bool] | None = IS_CHARACTER_JUNK,
    ) -> Iterator[Any]:
        kinds = _junk_kinds(linejunk, charjunk)
        # A context CPython cannot make a list of (an OverflowError there) runs its code.
        if (
            kinds is not None
            and (context is None or (type(context) is int and 0 <= context < sys.maxsize))
            and m.ndiff_ok()
        ):
            rows = _native._mdiff(
                fromlines, tolines, context, kinds[0], linejunk, kinds[1], _WINDOWED
            )
            if rows is not None:
                yield from rows
                return
        yield from fallback(fromlines, tolines, context, linejunk, charjunk)

    _mdiff.__doc__ = _std._mdiff.__doc__
    m.mdiff = _mdiff
    return _placed(_mdiff, m)


_mdiff = _make_mdiff(_OWN, _std_mdiff)


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


def _make_make_table(fallback: Any, hooks: dict[str, Any], m: _Module) -> Any:
    """`HtmlDiff.make_table` for module `m` (CPython's looks up `_mdiff` there)."""

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
            and -(2**31) <= tabsize < 2**31  # str.expandtabs takes a C int
            and (wrap is None or (type(wrap) is int and wrap >= 0))
            and type(numlines) is int
            and 0 <= numlines < sys.maxsize  # _mdiff makes a list of numlines + 1
            and all(getattr(cls, h) is f for h, f in hooks.items())
            and m.mdiff_ok()
        ):
            with_context = bool(context)
            try:
                rows = _native._html_prepare(
                    fromlines,
                    tolines,
                    tabsize,
                    min(wrap or 0, sys.maxsize),  # no line is that long: no wrapping
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
_std_make_prefix = _std.HtmlDiff.__dict__["_make_prefix"]
_OwnHtmlDiff = HtmlDiff


def _make_prefix(self: Any) -> None:
    # One counter for the tables of both modules, as CPython has one: difflib.HtmlDiff's, so no
    # two tables in a process share anchor ids. A value set on this class (to reset it, as
    # CPython's tests do) moves to that counter first.
    own = _OwnHtmlDiff.__dict__.get("_default_prefix")
    if own is not None:
        _std.HtmlDiff._default_prefix = own
        with contextlib.suppress(AttributeError):
            del _OwnHtmlDiff._default_prefix
    _std_make_prefix(self)


_make_prefix.__doc__ = _std_make_prefix.__doc__
_make_prefix.__qualname__ = "HtmlDiff._make_prefix"
HtmlDiff._make_prefix = _make_prefix  # type: ignore[method-assign]
HtmlDiff.make_table = _make_make_table(  # type: ignore[method-assign]
    HtmlDiff.__dict__["make_table"], {h: getattr(HtmlDiff, h) for h in _HTML_HOOKS}, _OWN
)
# What pyroid.patch("difflib") sets as difflib.HtmlDiff.make_table.
_PATCH_MAKE_TABLE = _make_make_table(
    _std.HtmlDiff.__dict__["make_table"],
    {h: _std.HtmlDiff.__dict__[h] for h in _HTML_HOOKS},
    _STD,
)

for _cls in (SequenceMatcher, Differ, HtmlDiff):
    _cls.__module__ = __name__

del _name, _f, _cls

#: What pyroid.patch("difflib") sets on the standard library's module: these functions with
#: difflib's globals (and difflib's own functions to fall back to).
_PATCH_FUNCTIONS: dict[str, Any] = {
    "get_close_matches": _make_close(_STD, _STD_FUNCTIONS["get_close_matches"]),
    "unified_diff": _make_unified(_STD, _STD_FUNCTIONS["unified_diff"]),
    "context_diff": _make_context(_STD, _STD_FUNCTIONS["context_diff"]),
    "_mdiff": _make_mdiff(_STD, _STD_FUNCTIONS["_mdiff"]),
}

# The fallbacks' and the patched functions' docstrings are the public functions' (their
# doctests run there once).
for _f in (
    _std_unified,
    _std_context,
    _std_close,
    _std_mdiff,
    _PATCH_COMPARE,
    _PATCH_MAKE_TABLE,
    *_PATCH_FUNCTIONS.values(),
):
    _f.__doc__ = None
del _f

if not _KNOWN:  # an interpreter whose difflib this module does not know: the standard library's
    SequenceMatcher = _std.SequenceMatcher  # type: ignore[misc]
    Differ = _std.Differ  # type: ignore[misc]
    HtmlDiff = _std.HtmlDiff  # type: ignore[misc]
    ndiff = _std.ndiff
    unified_diff = _std.unified_diff
    context_diff = _std.context_diff
    _mdiff = _std._mdiff

    def get_close_matches(  # type: ignore[misc]
        word: Sequence[Any],
        possibilities: Any,
        n: int = 3,
        cutoff: float = 0.6,
        *,
        threads: int | None = None,
    ) -> list[Any]:
        return _std.get_close_matches(word, possibilities, n, cutoff)  # threads: nothing to share

    get_close_matches.__doc__ = _std.get_close_matches.__doc__
    warnings.warn(
        f"pyroid.difflib reproduces difflib of CPython 3.11 to 3.15, not "
        f"{sys.version_info[0]}.{sys.version_info[1]}'s: it is the standard library's difflib "
        "here (slow path), and pyroid.patch('difflib') patches nothing. ratio_batch and "
        "close_matches_batch still run natively, with 3.15's SequenceMatcher.",
        FallbackWarning,
        stacklevel=2,
    )
