"""The python-dateutil objects pyroid's parser returns and raises (not public API).

With python-dateutil installed, these are its own: ``tz`` (so results carry ``tz.UTC``,
``tz.tzoffset(...)``, ``tz.tzlocal()``), ``relativedelta``, ``ParserError`` and
``UnknownTimezoneWarning`` (so ``except dateutil.parser.ParserError`` catches pyroid's errors).
Without it, pyroid's copies of dateutil 4.0.0's classes stand in (:mod:`pyroid.dateutil._tzlite`);
the few calls that need the rest of ``dateutil.tz`` (a ``str`` in ``tzinfos``) raise
:class:`pyroid.PyroidError`.
"""

from __future__ import annotations

import datetime
import sys
import types
from typing import Any

from pyroid._pyroid import PyroidError

__all__ = [
    "INSTALLED_VERSION",
    "ParserError",
    "UnknownTimezoneWarning",
    "relativedelta",
    "tz",
]


def _real(name: str) -> Any:
    """The real module `name`, or None (not installed, blocked, or only a pyroid shim)."""
    module = sys.modules.get(name)
    if module is not None and getattr(module, "__pyroid__", False):
        return module.__dict__.get("__wrapped__")
    try:
        __import__(name)
    except ImportError:
        return None
    module = sys.modules.get(name)
    if module is not None and getattr(module, "__pyroid__", False):
        return module.__dict__.get("__wrapped__")
    return module


def _installed_version() -> str | None:
    from importlib import metadata

    try:
        return metadata.version("python-dateutil")
    except metadata.PackageNotFoundError:
        return None


_tz = _real("dateutil.tz")
_parser_mod = _real("dateutil.parser._parser") if _tz is not None else None
_rd = _real("dateutil.relativedelta") if _tz is not None else None

#: The installed python-dateutil's version (None without it).
INSTALLED_VERSION: str | None = _installed_version() if _tz is not None else None

if _tz is not None and _rd is not None:
    tz: Any = _tz
    relativedelta: Any = _rd
else:
    from pyroid.dateutil import _tzlite

    class _TzStandIn(types.ModuleType):
        """``dateutil.tz`` for the parser, without python-dateutil."""

        UTC = _tzlite.UTC
        tzutc = _tzlite.tzutc
        tzoffset = _tzlite.tzoffset
        tzlocal = _tzlite.tzlocal
        enfold = staticmethod(_tzlite.enfold)

        @staticmethod
        def tzstr(s: str, posix_offset: bool = False) -> Any:
            raise PyroidError(
                f"pyroid.dateutil.parser: a str in tzinfos ({s!r}) needs dateutil.tz.tzstr, and "
                "python-dateutil is not installed (pip install python-dateutil)"
            )

    class _RelativeDeltaStandIn(types.ModuleType):
        """``dateutil.relativedelta`` for the parser's one use: ``relativedelta(weekday=n)``."""

        class relativedelta:  # noqa: N801 - dateutil's name
            def __init__(self, *, weekday: int) -> None:
                # dateutil indexes its weekday tuple (MO..SU): IndexError past it, negatives wrap.
                self.weekday = range(7)[weekday]

            def __radd__(self, other: datetime.datetime) -> datetime.datetime:
                return other + datetime.timedelta(days=(7 - other.weekday() + self.weekday) % 7)

    tz = _TzStandIn("dateutil.tz")
    relativedelta = _RelativeDeltaStandIn("dateutil.relativedelta")

_ParserError = getattr(_parser_mod, "ParserError", None)
_UnknownTimezoneWarning = getattr(_parser_mod, "UnknownTimezoneWarning", None)

if isinstance(_ParserError, type) and issubclass(_ParserError, ValueError):
    ParserError: type[ValueError] = _ParserError
else:

    class ParserError(ValueError):  # type: ignore[no-redef]
        """Exception subclass used for any failure to parse a datetime string.

        This is a subclass of :py:exc:`ValueError`, and should be raised any time
        earlier versions of ``dateutil`` would have raised ``ValueError``.

        .. versionadded:: 2.8.1
        """

        __module__ = "pyroid.dateutil.parser"

        def __str__(self) -> str:
            try:
                return self.args[0] % self.args[1:]  # type: ignore[no-any-return]
            except (TypeError, IndexError):
                return super().__str__()

        def __repr__(self) -> str:
            args = ", ".join("'%s'" % arg for arg in self.args)  # noqa: UP031 - dateutil's code
            return "%s(%s)" % (self.__class__.__name__, args)  # noqa: UP031


if isinstance(_UnknownTimezoneWarning, type) and issubclass(_UnknownTimezoneWarning, Warning):
    UnknownTimezoneWarning: type[Warning] = _UnknownTimezoneWarning
else:

    class UnknownTimezoneWarning(RuntimeWarning):  # type: ignore[no-redef]
        """Raised when the parser finds a timezone it cannot parse into a tzinfo.

        .. versionadded:: 2.7.0
        """

        __module__ = "pyroid.dateutil.parser"
