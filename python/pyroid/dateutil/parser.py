# Derived from python-dateutil 4.0.0 (BSD-3-Clause), (c) Gustavo Niemeyer, Paul Ganssle and
# dateutil contributors; modified. See pyroid's NOTICE and LICENSES/BSD-3-Clause-dateutil.txt.
"""A drop-in, Rust-backed ``dateutil.parser`` (``import pyroid.dateutil.parser``).

``parse``, ``parser``, ``parserinfo``, ``isoparse`` and ``isoparser`` have python-dateutil's
signatures and give the same results: the same ``datetime`` values, the installed dateutil's own
time zone objects (``tz.UTC``, ``tz.tzoffset(...)``, ``tz.tzlocal()``), and the same
``ParserError`` (dateutil's class, when it is installed) with the same message. ``parse`` is the
same in python-dateutil 2.8.2, 2.9.0, 2.9.0.post0, 3.0.0 and 4.0.0; ``isoparse`` follows the
installed release (see ``isoparser(compat=...)``). ``parse_batch`` and ``isoparse_batch`` parse
lists and Arrow columns on pyroid's thread pool.

The tokenizer and the parser's resolution logic run in Rust. A ``parser`` or ``parserinfo``
subclass that overrides a method, and the few inputs the Rust port does not handle, run dateutil's
own algorithm (pyroid's copy of it, :mod:`pyroid.dateutil._reference`), with a one-time
:class:`pyroid.FallbackWarning`. See ``parser.pyi``.
"""

from __future__ import annotations

import datetime
import sys
import unicodedata
import warnings
from collections.abc import Callable
from typing import Any

from pyroid._pyroid import FallbackWarning
from pyroid._pyroid import dateutil as _native
from pyroid.dateutil import _isoreference, _reference, _upstream
from pyroid.dateutil._upstream import ParserError, UnknownTimezoneWarning

__all__ = [
    "DATEUTIL_VERSION",
    "DATEUTIL_VERSIONS",
    "DEFAULTPARSER",
    "DEFAULTTZPARSER",
    "DEFAULT_ISOPARSER",
    "ParserError",
    "UnknownTimezoneWarning",
    "isoparse",
    "isoparse_batch",
    "isoparser",
    "parse",
    "parse_batch",
    "parser",
    "parserinfo",
]

#: The newest python-dateutil release this module reproduces.
DATEUTIL_VERSION = "4.0.0"
#: Every release it reproduces: ``parse`` is the same in all of them; ``isoparse`` has a 4.0.0
#: form and a 2.9.0.post0 form (which 2.8.2, 2.9.0 and 3.0.0 share), selected with
#: ``isoparser(compat=...)``.
DATEUTIL_VERSIONS = ("4.0.0", "3.0.0", "2.9.0.post0", "2.9.0", "2.8.2")

# isoparse modes: the release whose isoparse a compat value means. 2.8.2's and 2.9.0's
# isoparser.py and _parser.py are 2.9.0.post0's code (2.8.2's differs in one docstring line).
_ISO_MODES = {
    "4.0.0": "4.0.0",
    "3.0.0": "2.9.0.post0",
    "2.9.0.post0": "2.9.0.post0",
    "2.9.0": "2.9.0.post0",
    "2.8.2": "2.9.0.post0",
}


def _is_internal(frame: Any) -> bool:
    """Whether ``warnings.warn`` skips `frame` when it counts ``stacklevel`` (importlib's)."""
    filename = frame.f_code.co_filename
    return "importlib" in filename and "_bootstrap" in filename


def _stacklevel_outside(prefixes: tuple[str, ...]) -> int:
    """The ``stacklevel`` that points a warning at the first frame outside `prefixes`.

    For the caller's ``warnings.warn``: the first frame whose module name does not start with
    one of `prefixes`. ``warnings.warn`` does not count importlib's bootstrap frames, so neither
    does this.
    """
    frame = sys._getframe(1)
    level = 1
    while frame.f_back is not None:
        frame = frame.f_back
        if _is_internal(frame):
            continue
        level += 1
        if not frame.f_globals.get("__name__", "").startswith(prefixes):
            break
    return level


def _version_check() -> None:
    """Warn once if the installed python-dateutil is a release this module does not reproduce."""
    try:
        from pyroid import _versions  # type: ignore[attr-defined,unused-ignore]
    except ImportError:
        _versions = None
    if _versions is not None:
        _versions.check(
            __name__,
            "dateutil",
            f"python-dateutil=={DATEUTIL_VERSION}",
            same=tuple(f"python-dateutil=={v}" for v in DATEUTIL_VERSIONS[1:]),
        )
        return
    found = _upstream.INSTALLED_VERSION
    if found is None or found in DATEUTIL_VERSIONS:
        return
    others = ", ".join(DATEUTIL_VERSIONS[1:])
    warnings.warn(
        f"{__name__} reproduces python-dateutil {DATEUTIL_VERSION} (and {others}), but "
        f"python-dateutil {found} is installed: results can differ from it. Install one of "
        "those releases to compare like for like.",
        RuntimeWarning,
        stacklevel=_stacklevel_outside(("pyroid", "importlib")),
    )


_version_check()


def _unicode_version() -> str:
    version = unicodedata.unidata_version
    supported = _native.supported_unicode_versions()
    if version in supported:
        return version
    newest = supported[-1]
    warnings.warn(
        f"pyroid.dateutil has no tables for Unicode {version} (this Python's unicodedata); "
        f"using Unicode {newest}. Output can differ from dateutil for characters added "
        f"after {newest}.",
        RuntimeWarning,
        stacklevel=3,
    )
    return newest


_native._set_unicode_version(_unicode_version())

_WARNED: set[str] = set()


def _fallback(name: str, reason: str | None) -> None:
    """Warn once that dateutil's Python algorithm runs, for `reason`.

    `None` means no warning: the call's result or error is dateutil's anyway (an unexpected
    keyword argument, for one).
    """
    if reason is None:
        return
    key = f"{name}:{reason}"
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(
            f"pyroid.dateutil.parser.{name}: {reason} is not supported by the Rust port; "
            "running dateutil's own algorithm (slow path). This warning is shown once.",
            FallbackWarning,
            stacklevel=_stacklevel_outside(("pyroid.dateutil",)),
        )


# ---------------------------------------------------------------------------
# parserinfo, parser
# ---------------------------------------------------------------------------

#: dateutil's parserinfo class (pyroid's copy): subclass it to change the accepted words.
parserinfo = _reference.parserinfo
parserinfo.__module__ = __name__
_reference._changed = _native._info_changed

_DATEUTIL_PARSERINFO: Any = getattr(_upstream._parser_mod, "parserinfo", None)

# The parser methods the Rust port replaces; a subclass overriding one runs dateutil's code.
_PARSER_HOOKS = (
    "_parse",
    "_parse_numeric_token",
    "_find_hms_idx",
    "_assign_hms",
    "_could_be_tzname",
    "_ampm_valid",
    "_adjust_ampm",
    "_parse_min_sec",
    "_parse_hms",
    "_parsems",
    "_to_decimal",
    "_build_tzinfo",
    "_build_tzaware",
    "_build_naive",
    "_assign_tzname",
    "_recombine_skipped",
    "_result",
)
# The ones `_parse` calls (a subclass overriding `_parse` itself may call pyroid's).
_PRIVATE_HOOKS = tuple(
    h
    for h in _PARSER_HOOKS
    if h not in ("_parse", "_build_tzinfo", "_build_tzaware", "_build_naive", "_assign_tzname")
)
_INFO_HOOKS = (
    "jump",
    "weekday",
    "month",
    "hms",
    "ampm",
    "pertain",
    "utczone",
    "tzoffset",
    "convertyear",
    "validate",
    "_convert",
)


class parser(_reference.parser):  # noqa: N801 - dateutil's name
    """dateutil's ``parser``: ``parser(info=None).parse(timestr, ...)``.

    ``info`` is a :class:`parserinfo` (or dateutil's own). The parsing runs in Rust unless a
    subclass overrides one of the private methods dateutil documents as hooks, in which case
    dateutil's algorithm runs with them.
    """

    def __init__(self, info: Any = None) -> None:
        self.info = info or parserinfo()

    def parse(
        self,
        timestr: Any,
        default: datetime.datetime | None = None,
        ignoretz: bool = False,
        tzinfos: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Parse the date/time string into a :class:`datetime.datetime` (dateutil's ``parse``)."""
        return _native.parser_parse(self, timestr, default, ignoretz, tzinfos, kwargs)

    def _parse(
        self,
        timestr: Any,
        dayfirst: Any = None,
        yearfirst: Any = None,
        fuzzy: Any = False,
        fuzzy_with_tokens: Any = False,
    ) -> Any:
        return _native.parser_private_parse(
            self, timestr, dayfirst, yearfirst, fuzzy, fuzzy_with_tokens
        )


def _make_result(fields: tuple[Any, ...], century_specified: bool) -> Any:
    """A ``parser._result`` holding the kernel's fields (for dateutil's ``_build_*`` steps)."""
    res = parser._result()
    for attr, value in zip(parser._result.__slots__, fields, strict=False):
        setattr(res, attr, value)
    res.century_specified = century_specified
    return res


def _finish(
    self: Any,
    res: Any,
    skipped_tokens: Any,
    timestr: Any,
    default: Any,
    ignoretz: Any,
    tzinfos: Any,
    kwargs: dict[str, Any],
) -> Any:
    """The rest of dateutil's ``parser.parse`` after ``_parse`` (its exact code)."""
    if res is None:
        raise ParserError("Unknown string format: %s", timestr)
    if len(res) == 0:
        raise ParserError("String does not contain a date: %s", timestr)
    try:
        ret = self._build_naive(res, default)
    except ValueError as e:
        raise ParserError(str(e) + ": %s", timestr) from e
    if not ignoretz:
        ret = self._build_tzaware(ret, res, tzinfos)
    if kwargs.get("fuzzy_with_tokens", False):
        return ret, skipped_tokens
    return ret


def _reference_parse(
    self: Any,
    reason: str | None,
    timestr: Any,
    default: Any,
    ignoretz: Any,
    tzinfos: Any,
    kwargs: dict[str, Any],
) -> Any:
    """Run dateutil's ``parser.parse`` itself (pyroid's copy), after a FallbackWarning.

    `timestr` may be a :class:`_Replay` of a stream: dateutil's code reads it, and its errors
    name the caller's stream.
    """
    _fallback("parse", reason)
    # dateutil's parse. It calls self._parse, which for pyroid's own _parse is dateutil's _parse
    # here: pyroid's would run the kernel again and warn a second time for the same reason.
    if default is None:
        default = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    own = self._parse
    if getattr(own, "__func__", None) is parser._parse:
        res, skipped_tokens = _reference.parser._parse(self, timestr, **kwargs)
    else:
        res, skipped_tokens = own(timestr, **kwargs)
    if isinstance(timestr, _Replay):
        timestr = timestr._original
    return _finish(self, res, skipped_tokens, timestr, default, ignoretz, tzinfos, kwargs)


def _reference_private_parse(self: Any, reason: str, timestr: Any, *args: Any) -> Any:
    _fallback("parser._parse", reason)
    return _reference.parser._parse(self, timestr, *args)


class _Replay:
    """A stream that replays the ``read(1)`` results taken from another one, then reads on.

    ``str()`` and ``repr()`` are the original stream's, so dateutil's messages name it.
    """

    def __init__(self, chunks: list[Any], original: Any) -> None:
        self._chunks = chunks
        self._i = 0
        self._original = original

    def __str__(self) -> str:
        return str(self._original)

    def __repr__(self) -> str:
        return repr(self._original)

    def read(self, n: int = -1) -> Any:
        if self._i < len(self._chunks):
            self._i += 1
            return self._chunks[self._i - 1]
        return self._original.read(n)


def _replay_read(text: str, stream: Any) -> _Replay:
    """A replay of the reads :func:`_read_stream` took to read `text` from `stream`.

    One ``read(1)`` result per character, then the ``""`` that ended it: dateutil's code, run
    after the Rust parser read the stream, reads the same characters and does not read `stream`
    again.
    """
    return _Replay([*text, ""], stream)


def _read_stream(stream: Any) -> Any:
    """Read a character stream as ``_timelex`` does, one ``read(1)`` at a time.

    Returns its text, or, at the first read that is not a one-character ``str``, a replay of the
    reads so far that then reads on from `stream` (dateutil's own code reads it, and fails at
    that read as dateutil does, instead of this reading a stream that never ends).
    """
    chunks = []
    while True:
        c = stream.read(1)
        if type(c) is not str or len(c) > 1:
            return _Replay([*chunks, c], stream)
        if not c:  # ""
            return "".join(chunks)
        chunks.append(c)


DEFAULTPARSER = parser()


# ---------------------------------------------------------------------------
# isoparser
# ---------------------------------------------------------------------------


def _iso_mode(compat: str | None) -> str:
    if compat is None:
        installed = _upstream.INSTALLED_VERSION
        if installed is None:
            return DATEUTIL_VERSION
        if installed in _ISO_MODES:
            return _ISO_MODES[installed]
        # Another release (the import warned about it): the closer of the two forms.
        major = installed.partition(".")[0]
        return DATEUTIL_VERSION if major.isdigit() and int(major) >= 4 else "2.9.0.post0"
    if compat not in _ISO_MODES:
        raise ValueError(
            f"compat must be one of {', '.join(map(repr, DATEUTIL_VERSIONS))} or None, "
            f"not {compat!r}"
        )
    return _ISO_MODES[compat]


class isoparser(_isoreference.isoparser):  # noqa: N801 - dateutil's name
    """dateutil's ``isoparser(sep=None)``, plus the keyword-only ``compat``.

    ``compat`` picks whose ``isoparse`` to reproduce: ``"4.0.0"`` (each field exactly N ASCII
    digits), or ``"3.0.0"``, ``"2.9.0.post0"``, ``"2.9.0"`` or ``"2.8.2"`` (fields read with
    ``int()``, so ``2014- 4-21`` parses). ``None``, the default, follows the installed
    python-dateutil, and is 4.0.0 without it.
    """

    def __init__(self, sep: str | None = None, *, compat: str | None = None) -> None:
        super().__init__(sep)
        self._compat = _iso_mode(compat)
        self._lenient = self._compat != "4.0.0"

    @_isoreference._takes_ascii
    def isoparse(self, dt_str: Any) -> datetime.datetime:
        """Parse an ISO-8601 datetime string into a :class:`datetime.datetime`."""
        return _native.isoparser_isoparse(self, dt_str)  # type: ignore[no-any-return]


_isoparser_isoparse_doc = _isoreference.isoparser.isoparse.__doc__
isoparser.isoparse.__doc__ = _isoparser_isoparse_doc

# The isoparser methods the Rust port replaces.
_ISO_HOOKS = (
    "_parse_isodate",
    "_parse_isodate_common",
    "_parse_isodate_uncommon",
    "_calculate_weekdate",
    "_parse_isotime",
    "_parse_tzstr",
    "_parse_digits",
)

DEFAULT_ISOPARSER = isoparser()


def _reference_isoparse(self: Any, reason: str | None, dt_str: bytes) -> Any:
    """Run dateutil's ``isoparser.isoparse`` (pyroid's copy) on the ASCII bytes."""
    if reason is not None:
        _fallback("isoparse", reason)
    return _isoreference.isoparser.isoparse.__wrapped__(self, dt_str)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# The native module's hooks into this one
# ---------------------------------------------------------------------------

_native._install(
    {
        "module": sys.modules[__name__].__dict__,
        "parser": parser,
        "parserinfo": parserinfo,
        "dateutil_parserinfo": _DATEUTIL_PARSERINFO,
        "parser_hooks": _PARSER_HOOKS,
        "private_hooks": _PRIVATE_HOOKS,
        "info_hooks": _INFO_HOOKS,
        "iso_hooks": _ISO_HOOKS,
        "isoparser": isoparser,
        "ParserError": ParserError,
        "UnknownTimezoneWarning": UnknownTimezoneWarning,
        "tz": _upstream.tz,
        "finish": _finish,
        "make_result": _make_result,
        "reference_parse": _reference_parse,
        "reference_private_parse": _reference_private_parse,
        "reference_isoparse": _reference_isoparse,
        "read_stream": _read_stream,
        "replay_read": _replay_read,
        "datetime_module": datetime,
    }
)

#: ``parse(timestr, parserinfo=None, **kwargs)``: dateutil's module-level ``parse``.
parse: Callable[..., Any] = _native.parse
#: ``isoparse(dt_str)``: ``DEFAULT_ISOPARSER.isoparse``.
isoparse: Callable[..., datetime.datetime] = _native.isoparse
# pickle finds a function by its __module__ and name: this module (the native one is not
# importable), so Pool.map(parse, ...) works.
parse.__module__ = isoparse.__module__ = __name__


def _patched_functions() -> tuple[Callable[..., Any], Callable[..., datetime.datetime]]:
    """``parse`` and ``isoparse`` for ``pyroid.patch("dateutil")`` to set on dateutil's modules.

    They are dateutil's module-level functions (its ``DEFAULTPARSER``, its ``parser`` class, its
    ``DEFAULT_ISOPARSER``), parsed by pyroid; the installed release decides the isoparse mode.
    """
    import importlib

    du_parser = importlib.import_module("dateutil.parser._parser")
    du_iso = importlib.import_module("dateutil.parser.isoparser")
    _native._install_upstream(
        {
            "module": du_parser.__dict__,
            "parser": du_parser.parser,
            "iso_module": du_iso.__dict__,
            "isoparser": du_iso.isoparser,
            "lenient": _iso_mode(None) != "4.0.0",
        }
    )
    up_parse, up_isoparse = _native.upstream.parse, _native.upstream.isoparse
    # Where dateutil's own functions live, so pickle finds them there (pyroid's while patched,
    # dateutil's in a process that is not).
    up_parse.__module__ = du_parser.__name__
    up_isoparse.__module__ = du_iso.__name__
    return up_parse, up_isoparse


def parse_batch(
    timestrs: Any,
    parserinfo: Any = None,
    *,
    threads: int | None = None,
    errors: str = "raise",
    **kwargs: Any,
) -> Any:
    """:func:`parse` over many strings at once, on pyroid's thread pool (not part of dateutil).

    See ``parser.pyi``.
    """
    return _native.parse_batch(timestrs, parserinfo, threads, errors, kwargs)


def isoparse_batch(
    dt_strs: Any,
    *,
    threads: int | None = None,
    errors: str = "raise",
    sep: str | None = None,
    compat: str | None = None,
) -> Any:
    """:func:`isoparse` over many strings at once, on pyroid's thread pool (not part of dateutil).

    See ``parser.pyi``.
    """
    p = DEFAULT_ISOPARSER if sep is None and compat is None else isoparser(sep, compat=compat)
    return _native.isoparse_batch(dt_strs, p, threads, errors)


# dateutil.parser's private names that code outside it uses (pandas imports _timelex).
_timelex = _reference._timelex
_resultbase = _reference._resultbase
_tzparser = _reference._tzparser
_parsetz = _reference._parsetz
#: dateutil's ``DEFAULTTZPARSER`` (the ``TZ``-string parser behind ``dateutil.tz.tzstr``).
DEFAULTTZPARSER = _reference.DEFAULTTZPARSER
