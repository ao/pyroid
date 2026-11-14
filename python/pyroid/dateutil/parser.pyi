"""A drop-in, Rust-backed ``dateutil.parser`` (``import pyroid.dateutil.parser``).

``parse``, ``parser``, ``parserinfo``, ``isoparse`` and ``isoparser`` take python-dateutil's
arguments and give its results: the same ``datetime`` values, the installed dateutil's own time
zone objects, and the same exceptions with the same messages. ``parse`` is the same in
python-dateutil 2.9.0.post0, 3.0.0 and 4.0.0. ``isoparse`` follows the installed release
(``isoparser(compat=...)``).

``parse_batch`` and ``isoparse_batch`` are pyroid's additions: they parse lists, tuples and Arrow
string columns on pyroid's thread pool.
"""

import datetime
from collections.abc import Callable, Mapping
from typing import IO, Any, Final, Literal, TypeAlias, overload

from pyroid._pyroid import Column
from pyroid._typing import ArrowStrings

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

DATEUTIL_VERSION: Final[str]
"""The newest python-dateutil release this module reproduces (``"4.0.0"``)."""

DATEUTIL_VERSIONS: Final[tuple[str, ...]]
"""Every release it reproduces: ``("4.0.0", "3.0.0", "2.9.0.post0")``."""

_TzData: TypeAlias = datetime.tzinfo | str | int | None
_TzInfos: TypeAlias = Mapping[str, _TzData] | Callable[[str | None, int | None], _TzData]
_TimeStr: TypeAlias = str | bytes | bytearray | IO[str]
_Compat: TypeAlias = Literal["4.0.0", "3.0.0", "2.9.0.post0"]

class ParserError(ValueError):
    """dateutil's ``ParserError``: any failure to parse a date string.

    With python-dateutil installed this is ``dateutil.parser.ParserError`` itself, so code that
    catches dateutil's exception catches pyroid's. ``str(e)`` is dateutil's message, for example
    ``"Unknown string format: 32/13/2020"``.
    """

class UnknownTimezoneWarning(RuntimeWarning):
    """A time zone name was found but not understood (pass ``tzinfos``).

    dateutil's ``UnknownTimezoneWarning`` itself when python-dateutil is installed.
    """

class parserinfo:  # noqa: N801 - dateutil's name
    """The words the parser accepts. Subclass it and change the lists to parse other languages.

    The lists (``JUMP``, ``WEEKDAYS``, ``MONTHS``, ``HMS``, ``AMPM``, ``UTCZONE``, ``PERTAIN``,
    ``TZOFFSET``) are read by the Rust parser. A subclass that overrides a method instead runs
    dateutil's own algorithm, with a one-time :class:`pyroid.FallbackWarning`.

    Args:
        dayfirst: Read an ambiguous ``01/05/09`` as day first.
        yearfirst: Read an ambiguous ``01/05/09`` as year first.
    """

    JUMP: list[str]
    WEEKDAYS: list[tuple[str, ...]]
    MONTHS: list[tuple[str, ...]]
    HMS: list[tuple[str, ...]]
    AMPM: list[tuple[str, ...]]
    UTCZONE: list[str]
    PERTAIN: list[str]
    TZOFFSET: dict[str, int]
    dayfirst: bool
    yearfirst: bool
    def __init__(self, dayfirst: bool = False, yearfirst: bool = False) -> None: ...
    def jump(self, name: str) -> bool:
        """Whether ``name`` is a word to skip (``"at"``, ``"of"``, punctuation)."""
    def weekday(self, name: str) -> int | None:
        """The weekday number (Monday is 0) ``name`` stands for."""
    def month(self, name: str) -> int | None:
        """The month number (January is 1) ``name`` stands for."""
    def hms(self, name: str) -> int | None:
        """0, 1 or 2 if ``name`` marks hours, minutes or seconds."""
    def ampm(self, name: str) -> int | None:
        """0 for am, 1 for pm."""
    def pertain(self, name: str) -> bool:
        """Whether ``name`` is ``"of"`` (as in ``Jan of 01``)."""
    def utczone(self, name: str) -> bool:
        """Whether ``name`` names UTC."""
    def tzoffset(self, name: str) -> int | None:
        """The offset in seconds of time zone name ``name``, from ``TZOFFSET``."""
    def convertyear(self, year: int, century_specified: bool = False) -> int:
        """A two-digit year within 50 years of the year the parserinfo was made in."""
    def validate(self, res: Any) -> bool:
        """Normalise the time zone fields of a parse result (dateutil's hook)."""

class parser:  # noqa: N801 - dateutil's name
    """dateutil's ``parser``: ``parser(info).parse(timestr, ...)``.

    Args:
        info: A :class:`parserinfo` (or dateutil's), the default one if ``None``.
    """

    info: parserinfo
    def __init__(self, info: parserinfo | None = None) -> None: ...
    @overload
    def parse(
        self,
        timestr: _TimeStr,
        default: datetime.datetime | None = None,
        ignoretz: bool = False,
        tzinfos: _TzInfos | None = None,
        *,
        dayfirst: bool | None = ...,
        yearfirst: bool | None = ...,
        fuzzy: bool = ...,
        fuzzy_with_tokens: Literal[False] = False,
    ) -> datetime.datetime: ...
    @overload
    def parse(
        self,
        timestr: _TimeStr,
        default: datetime.datetime | None = None,
        ignoretz: bool = False,
        tzinfos: _TzInfos | None = None,
        *,
        dayfirst: bool | None = ...,
        yearfirst: bool | None = ...,
        fuzzy: bool = ...,
        fuzzy_with_tokens: Literal[True],
    ) -> tuple[datetime.datetime, tuple[str, ...]]: ...

DEFAULTPARSER: parser
"""The parser behind :func:`parse`."""

DEFAULTTZPARSER: Any
"""dateutil's ``DEFAULTTZPARSER``: the ``TZ`` string parser behind ``dateutil.tz.tzstr``."""

@overload
def parse(
    timestr: _TimeStr,
    parserinfo: parserinfo | None = None,
    *,
    default: datetime.datetime | None = ...,
    ignoretz: bool = ...,
    tzinfos: _TzInfos | None = ...,
    dayfirst: bool | None = ...,
    yearfirst: bool | None = ...,
    fuzzy: bool = ...,
    fuzzy_with_tokens: Literal[False] = False,
) -> datetime.datetime:
    """Parse a date/time string, as ``dateutil.parser.parse`` does.

    >>> parse("Thu Sep 25 10:36:28 2003")
    datetime.datetime(2003, 9, 25, 10, 36, 28)

    Args:
        timestr: The string (or bytes, or a text stream).
        parserinfo: The accepted words; dateutil's defaults if ``None``.
        default: The ``datetime`` whose fields fill in what the string does not give; today at
            midnight if ``None``.
        ignoretz: Return a naive ``datetime``, ignoring any time zone in the string.
        tzinfos: Time zone names (and offsets) to ``tzinfo`` objects: a dict of name to
            ``tzinfo``, ``TZ`` string or offset in seconds, or a function ``(name, offset)``.
        dayfirst: Read an ambiguous ``01/05/09`` as day first.
        yearfirst: Read an ambiguous ``01/05/09`` as year first.
        fuzzy: Skip words that are not part of a date (``"Today is January 1, 2047"``).
        fuzzy_with_tokens: As ``fuzzy``, and also return the skipped parts.

    Returns:
        The ``datetime``; with ``fuzzy_with_tokens``, ``(datetime, skipped_tokens)``.

    Raises:
        ParserError: The string is not a date, or gives an invalid one.
        OverflowError: A value exceeds the largest C integer.
        TypeError: ``timestr`` is not a string or stream.
    """

@overload
def parse(
    timestr: _TimeStr,
    parserinfo: parserinfo | None = None,
    *,
    default: datetime.datetime | None = ...,
    ignoretz: bool = ...,
    tzinfos: _TzInfos | None = ...,
    dayfirst: bool | None = ...,
    yearfirst: bool | None = ...,
    fuzzy: bool = ...,
    fuzzy_with_tokens: Literal[True],
) -> tuple[datetime.datetime, tuple[str, ...]]: ...

class isoparser:  # noqa: N801 - dateutil's name
    """dateutil's ``isoparser(sep=None)``, plus the keyword-only ``compat``.

    Args:
        sep: The one character that must separate date and time (``"T"`` for strict ISO 8601);
            any character if ``None``.
        compat: Whose ``isoparse`` to reproduce. ``"4.0.0"``: every numeric field is exactly N
            ASCII digits. ``"2.9.0.post0"`` and ``"3.0.0"``: fields are read with ``int()``, so
            ``"2014- 4-21"`` parses. ``None`` (the default): the installed python-dateutil's
            release, and 4.0.0 without one.

    Raises:
        ValueError: ``sep`` is not a single non-digit ASCII character, or ``compat`` is not one
            of :data:`DATEUTIL_VERSIONS`.
    """

    def __init__(self, sep: str | None = None, *, compat: _Compat | None = None) -> None: ...
    def isoparse(self, dt_str: str | bytes | IO[str] | IO[bytes]) -> datetime.datetime:
        """Parse an ISO 8601 date and time string (dateutil's ``isoparser.isoparse``).

        Raises:
            ValueError: The string is not ISO 8601 in a form dateutil accepts.
        """
    def parse_isodate(self, datestr: str | bytes) -> datetime.date:
        """Parse the date part of an ISO 8601 string."""
    def parse_isotime(self, timestr: str | bytes) -> datetime.time:
        """Parse the time part of an ISO 8601 string."""
    def parse_tzstr(self, tzstr: str | bytes, zero_as_utc: bool = True) -> datetime.tzinfo:
        """Parse an ISO 8601 UTC offset (``Z``, ``+HH:MM``, ``+HHMM``, ``+HH``)."""

DEFAULT_ISOPARSER: isoparser
"""The isoparser behind :func:`isoparse` (``compat=None``)."""

def isoparse(dt_str: str | bytes | IO[str] | IO[bytes]) -> datetime.datetime:
    """Parse an ISO 8601 date and time string, as ``dateutil.parser.isoparse`` does.

    >>> isoparse("2003-09-25T10:49:41+03:00")
    datetime.datetime(2003, 9, 25, 10, 49, 41, tzinfo=tzoffset(None, 10800))

    Raises:
        ValueError: The string is not ISO 8601 in a form dateutil accepts.
    """

_Errors: TypeAlias = Literal["raise", "coerce"]

@overload
def parse_batch(
    timestrs: str,
    parserinfo: parserinfo | None = None,
    *,
    threads: int | None = None,
    errors: _Errors = "raise",
    default: datetime.datetime | None = ...,
    ignoretz: bool = ...,
    tzinfos: _TzInfos | None = ...,
    dayfirst: bool | None = ...,
    yearfirst: bool | None = ...,
    fuzzy: bool = ...,
) -> datetime.datetime | None:
    """:func:`parse` over many strings at once, on pyroid's thread pool.

    Not part of dateutil. Each result is what ``parse(s, ...)`` gives, except that
    ``default=None`` means today at midnight as read once for the batch.

    >>> parse_batch(["2003-09-25", "Sep 26 2003 10:00"])
    [datetime.datetime(2003, 9, 25, 0, 0), datetime.datetime(2003, 9, 26, 10, 0)]

    Args:
        timestrs: A str, a list or tuple of str, or an Arrow string array or chunked array.
        parserinfo: As for :func:`parse`.
        threads: At most this many tasks at once on the shared pool (``None``: the whole pool).
        errors: ``"raise"`` raises the first failing string's exception (with a note naming its
            index); ``"coerce"`` gives ``None`` for the strings that raise ``ValueError`` (such
            as ``ParserError``) or ``OverflowError``.
        default: As for :func:`parse`.
        ignoretz: As for :func:`parse`.
        tzinfos: As for :func:`parse`.
        dayfirst: As for :func:`parse`.
        yearfirst: As for :func:`parse`.
        fuzzy: As for :func:`parse`. ``fuzzy_with_tokens`` is not accepted.

    Returns:
        The results in the input's kind: a ``datetime`` (or ``None``) for a str, a list for a
        list, a tuple for a tuple. Arrow input gives a :class:`pyroid.Column` of
        ``timestamp[us]``: wall time with no time zone when every result is naive, UTC instants
        (``tz="UTC"``) when every result is aware; Arrow nulls and coerced failures are nulls.

    Raises:
        ParserError: A string is not a date (``errors="raise"``).
        ValueError: Arrow input whose results mix naive and aware datetimes.
    """

@overload
def parse_batch(
    timestrs: list[str],
    parserinfo: parserinfo | None = None,
    *,
    threads: int | None = None,
    errors: _Errors = "raise",
    default: datetime.datetime | None = ...,
    ignoretz: bool = ...,
    tzinfos: _TzInfos | None = ...,
    dayfirst: bool | None = ...,
    yearfirst: bool | None = ...,
    fuzzy: bool = ...,
) -> list[datetime.datetime | None]: ...
@overload
def parse_batch(
    timestrs: tuple[str, ...],
    parserinfo: parserinfo | None = None,
    *,
    threads: int | None = None,
    errors: _Errors = "raise",
    default: datetime.datetime | None = ...,
    ignoretz: bool = ...,
    tzinfos: _TzInfos | None = ...,
    dayfirst: bool | None = ...,
    yearfirst: bool | None = ...,
    fuzzy: bool = ...,
) -> tuple[datetime.datetime | None, ...]: ...
@overload
def parse_batch(
    timestrs: ArrowStrings,
    parserinfo: parserinfo | None = None,
    *,
    threads: int | None = None,
    errors: _Errors = "raise",
    default: datetime.datetime | None = ...,
    ignoretz: bool = ...,
    tzinfos: _TzInfos | None = ...,
    dayfirst: bool | None = ...,
    yearfirst: bool | None = ...,
    fuzzy: bool = ...,
) -> Column[datetime.datetime]: ...
@overload
def isoparse_batch(
    dt_strs: str,
    *,
    threads: int | None = None,
    errors: _Errors = "raise",
    sep: str | None = None,
    compat: _Compat | None = None,
) -> datetime.datetime | None:
    """:func:`isoparse` over many strings at once, on pyroid's thread pool.

    Not part of dateutil. Each result is what ``isoparser(sep, compat=compat).isoparse(s)``
    gives.

    Args:
        dt_strs: A str, a list or tuple of str, or an Arrow string array or chunked array.
        threads: At most this many tasks at once on the shared pool (``None``: the whole pool).
        errors: ``"raise"`` or ``"coerce"``, as for :func:`parse_batch`.
        sep: As for :class:`isoparser`.
        compat: As for :class:`isoparser`.

    Returns:
        The results in the input's kind, as for :func:`parse_batch`.

    Raises:
        ValueError: A string is not ISO 8601 (``errors="raise"``), or Arrow input whose results
            mix naive and aware datetimes.
    """

@overload
def isoparse_batch(
    dt_strs: list[str],
    *,
    threads: int | None = None,
    errors: _Errors = "raise",
    sep: str | None = None,
    compat: _Compat | None = None,
) -> list[datetime.datetime | None]: ...
@overload
def isoparse_batch(
    dt_strs: tuple[str, ...],
    *,
    threads: int | None = None,
    errors: _Errors = "raise",
    sep: str | None = None,
    compat: _Compat | None = None,
) -> tuple[datetime.datetime | None, ...]: ...
@overload
def isoparse_batch(
    dt_strs: ArrowStrings,
    *,
    threads: int | None = None,
    errors: _Errors = "raise",
    sep: str | None = None,
    compat: _Compat | None = None,
) -> Column[datetime.datetime]: ...
