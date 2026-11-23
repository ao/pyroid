# Derived from python-dateutil 4.0.0 (BSD-3-Clause), (c) Gustavo Niemeyer, Paul Ganssle and
# dateutil contributors; modified. See pyroid's NOTICE and LICENSES/BSD-3-Clause-dateutil.txt.
"""dateutil 4.0.0's ``tzutc``, ``tzoffset``, ``tzlocal`` and ``enfold``, copied for interpreters
without python-dateutil (not public API).

:mod:`pyroid.dateutil.parser` returns the installed dateutil's own time zone objects when it is
installed. Without it, these copies give objects with the same behaviour, ``repr`` and equality.
"""

import _thread
import datetime
import time
import weakref
from collections import OrderedDict
from datetime import timedelta, tzinfo
from functools import wraps

__all__ = ["UTC", "enfold", "tzlocal", "tzoffset", "tzutc"]

# ---- dateutil/tz/_common.py




ZERO = timedelta(0)



# The following is adapted from Alexander Belopolsky's tz library
# https://github.com/abalkin/tz
def enfold(dt: datetime.datetime, fold: int = 1) -> datetime.datetime:
    """Returns a datetime with the ``fold`` attribute set to ``fold``.

    :param fold:
        The value for the ``fold`` attribute, either 0 or 1.

    :return:
        A :class:`datetime.datetime` with the requested ``fold`` value.

    .. versionadded:: 2.6.0
    """
    return dt.replace(fold=fold)


def _validate_fromutc_inputs(f):
    """
    The CPython version of ``fromutc`` checks that the input is a ``datetime``
    object and that ``self`` is attached as its ``tzinfo``.
    """
    @wraps(f)
    def fromutc(self, dt):
        if not isinstance(dt, datetime.datetime):
            raise TypeError("fromutc() requires a datetime argument")
        if dt.tzinfo is not self:
            raise ValueError("dt.tzinfo is not self")

        return f(self, dt)

    return fromutc


class _tzinfo(tzinfo):
    """
    Base class for all ``dateutil`` ``tzinfo`` objects.
    """

    def is_ambiguous(self, dt):
        """
        Whether or not the "wall time" of a given datetime is ambiguous in this
        zone.

        :param dt:
            A :py:class:`datetime.datetime`, naive or time zone aware.


        :return:
            Returns ``True`` if ambiguous, ``False`` otherwise.

        .. versionadded:: 2.6.0
        """

        dt = dt.replace(tzinfo=self)

        wall_0 = enfold(dt, fold=0)
        wall_1 = enfold(dt, fold=1)

        same_offset = wall_0.utcoffset() == wall_1.utcoffset()
        same_dt = wall_0.replace(tzinfo=None) == wall_1.replace(tzinfo=None)

        return same_dt and not same_offset

    def _fold_status(self, dt_utc, dt_wall):
        """
        Determine the fold status of a "wall" datetime, given a representation
        of the same datetime as a (naive) UTC datetime. This is calculated based
        on the assumption that ``dt.utcoffset() - dt.dst()`` is constant for all
        datetimes, and that this offset is the actual number of hours separating
        ``dt_utc`` and ``dt_wall``.

        :param dt_utc:
            Representation of the datetime as UTC

        :param dt_wall:
            Representation of the datetime as "wall time". This parameter must
            either have a `fold` attribute or have a fold-naive
            :class:`datetime.tzinfo` attached, otherwise the calculation may
            fail.
        """
        if self.is_ambiguous(dt_wall):
            delta_wall = dt_wall - dt_utc
            _fold = int(delta_wall == (dt_utc.utcoffset() - dt_utc.dst()))
        else:
            _fold = 0

        return _fold

    def _fromutc(self, dt):
        """
        Given a timezone-aware datetime in a given timezone, calculates a
        timezone-aware datetime in a new timezone.

        Since this is the one time that we *know* we have an unambiguous
        datetime object, we take this opportunity to determine whether the
        datetime is ambiguous and in a "fold" state (e.g. if it's the first
        occurrence, chronologically, of the ambiguous datetime).

        :param dt:
            A timezone-aware :class:`datetime.datetime` object.
        """

        # Re-implement the algorithm from Python's datetime.py
        dtoff = dt.utcoffset()
        if dtoff is None:
            raise ValueError("fromutc() requires a non-None utcoffset() "
                             "result")

        # The original datetime.py code assumes that `dst()` defaults to
        # zero during ambiguous times. PEP 495 inverts this presumption, so
        # select fold=1 when calculating the DST adjustment below.
        dtdst = dt.dst()
        if dtdst is None:
            raise ValueError("fromutc() requires a non-None dst() result")
        delta = dtoff - dtdst

        dt += delta
        # Set fold=1 so we can default to being in the fold for
        # ambiguous dates.
        dtdst = enfold(dt, fold=1).dst()
        if dtdst is None:
            raise ValueError("fromutc(): dt.dst gave inconsistent "
                             "results; cannot convert")
        return dt + dtdst

    @_validate_fromutc_inputs
    def fromutc(self, dt):
        """
        Given a timezone-aware datetime in a given timezone, calculates a
        timezone-aware datetime in a new timezone.

        Since this is the one time that we *know* we have an unambiguous
        datetime object, we take this opportunity to determine whether the
        datetime is ambiguous and in a "fold" state (e.g. if it's the first
        occurrence, chronologically, of the ambiguous datetime).

        :param dt:
            A timezone-aware :class:`datetime.datetime` object.
        """
        dt_wall = self._fromutc(dt)

        # Calculate the fold status given the two datetimes.
        _fold = self._fold_status(dt, dt_wall)

        # Set the default fold value for ambiguous dates
        return enfold(dt_wall, fold=_fold)


# ---- dateutil/tz/_factories.py


class _TzSingleton(type):
    def __init__(cls, *args, **kwargs):
        cls.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(cls):
        if cls.__instance is None:
            cls.__instance = super().__call__()
        return cls.__instance


class _TzFactory(type):
    def instance(cls, *args, **kwargs):
        """Alternate constructor that returns a fresh instance"""
        return type.__call__(cls, *args, **kwargs)


class _TzOffsetFactory(_TzFactory):
    def __init__(cls, *args, **kwargs):
        cls.__instances = weakref.WeakValueDictionary()
        cls.__strong_cache = OrderedDict()
        cls.__strong_cache_size = 8

        cls._cache_lock = _thread.allocate_lock()

    def __call__(cls, name, offset):
        if isinstance(offset, timedelta):
            key = (name, offset.total_seconds())
        else:
            key = (name, offset)

        instance = cls.__instances.get(key, None)
        if instance is None:
            instance = cls.__instances.setdefault(key,
                                                  cls.instance(name, offset))

        # This lock may not be necessary in Python 3. See GH issue #901
        with cls._cache_lock:
            cls.__strong_cache[key] = cls.__strong_cache.pop(key, instance)

            # Remove an item if the strong cache is overpopulated
            if len(cls.__strong_cache) > cls.__strong_cache_size:
                cls.__strong_cache.popitem(last=False)

        return instance

# ---- dateutil/tz/tz.py
ZERO = datetime.timedelta(0)
EPOCH = datetime.datetime(1970, 1, 1, 0, 0)


class tzutc(datetime.tzinfo, metaclass=_TzSingleton):
    """
    This is a tzinfo object that represents the UTC time zone.

    **Examples:**

    .. doctest::

        >>> from datetime import *
        >>> from dateutil.tz import *

        >>> datetime.now()
        datetime.datetime(2003, 9, 27, 9, 40, 1, 521290)

        >>> datetime.now(tzutc())
        datetime.datetime(2003, 9, 27, 12, 40, 12, 156379, tzinfo=tzutc())

        >>> datetime.now(tzutc()).tzname()
        'UTC'

    .. versionchanged:: 2.7.0
        ``tzutc()`` is now a singleton, so the result of ``tzutc()`` will
        always return the same object.

        .. doctest::

            >>> from dateutil.tz import tzutc, UTC
            >>> tzutc() is tzutc()
            True
            >>> tzutc() is UTC
            True
    """
    def utcoffset(self, dt):
        return ZERO

    def dst(self, dt):
        return ZERO

    def tzname(self, dt):
        return "UTC"

    def is_ambiguous(self, dt):
        """
        Whether or not the "wall time" of a given datetime is ambiguous in this
        zone.

        :param dt:
            A :py:class:`datetime.datetime`, naive or time zone aware.


        :return:
            Returns ``True`` if ambiguous, ``False`` otherwise.

        .. versionadded:: 2.6.0
        """
        return False

    @_validate_fromutc_inputs
    def fromutc(self, dt):
        """
        Fast track version of fromutc() returns the original ``dt`` object for
        any valid :py:class:`datetime.datetime` object.
        """
        return dt

    def __eq__(self, other):
        if not isinstance(other, (tzutc, tzoffset)):
            return NotImplemented

        return (isinstance(other, tzutc) or
                (isinstance(other, tzoffset) and other._offset == ZERO))

    __hash__ = None

    def __repr__(self):
        return "%s()" % self.__class__.__name__

    def __reduce__(self):
        return (self.__class__, tuple())


#: Convenience constant providing a :class:`tzutc()` instance
#:
#: .. versionadded:: 2.7.0
UTC = tzutc()


class tzoffset(datetime.tzinfo, metaclass=_TzOffsetFactory):
    """
    A simple class for representing a fixed offset from UTC.

    :param name:
        The timezone name, to be returned when ``tzname()`` is called.
    :param offset:
        The time zone offset in seconds, or (since version 2.6.0, represented
        as a :py:class:`datetime.timedelta` object).
    """
    def __init__(self, name, offset):
        self._name = name

        try:
            # Allow a timedelta
            offset = offset.total_seconds()
        except (TypeError, AttributeError):
            pass

        self._offset = datetime.timedelta(seconds=offset)

    def utcoffset(self, dt):
        return self._offset

    def dst(self, dt):
        return ZERO

    def tzname(self, dt):
        return self._name

    @_validate_fromutc_inputs
    def fromutc(self, dt):
        return dt + self._offset

    def is_ambiguous(self, dt):
        """
        Whether or not the "wall time" of a given datetime is ambiguous in this
        zone.

        :param dt:
            A :py:class:`datetime.datetime`, naive or time zone aware.
        :return:
            Returns ``True`` if ambiguous, ``False`` otherwise.

        .. versionadded:: 2.6.0
        """
        return False

    def __eq__(self, other):
        if not isinstance(other, tzoffset):
            return NotImplemented

        return self._offset == other._offset

    __hash__ = None

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               repr(self._name),
                               int(self._offset.total_seconds()))

    def __reduce__(self):
        return (self.__class__, (self._name, self._offset))


class tzlocal(_tzinfo):
    """
    A :class:`tzinfo` subclass built around the ``time`` timezone functions.
    """
    def __init__(self):
        super().__init__()

        self._std_offset = datetime.timedelta(seconds=-time.timezone)
        if time.daylight:
            self._dst_offset = datetime.timedelta(seconds=-time.altzone)
        else:
            self._dst_offset = self._std_offset

        self._dst_saved = self._dst_offset - self._std_offset
        self._hasdst = bool(self._dst_saved)
        self._tznames = tuple(time.tzname)

    def utcoffset(self, dt):
        if dt is None and self._hasdst:
            return None

        if self._isdst(dt):
            return self._dst_offset
        else:
            return self._std_offset

    def dst(self, dt):
        if dt is None and self._hasdst:
            return None

        if self._isdst(dt):
            return self._dst_offset - self._std_offset
        else:
            return ZERO

    def tzname(self, dt):
        return self._tznames[self._isdst(dt)]

    def is_ambiguous(self, dt):
        """
        Whether or not the "wall time" of a given datetime is ambiguous in this
        zone.

        :param dt:
            A :py:class:`datetime.datetime`, naive or time zone aware.


        :return:
            Returns ``True`` if ambiguous, ``False`` otherwise.

        .. versionadded:: 2.6.0
        """
        naive_dst = self._naive_is_dst(dt)
        return (not naive_dst and
                (naive_dst != self._naive_is_dst(dt - self._dst_saved)))

    def _naive_is_dst(self, dt):
        timestamp = _datetime_to_timestamp(dt)
        return time.localtime(timestamp + time.timezone).tm_isdst

    def _isdst(self, dt, fold_naive=True):
        # We can't use mktime here. It is unstable when deciding if
        # the hour near to a change is DST or not.
        #
        # timestamp = time.mktime((dt.year, dt.month, dt.day, dt.hour,
        #                         dt.minute, dt.second, dt.weekday(), 0, -1))
        # return time.localtime(timestamp).tm_isdst
        #
        # The code above yields the following result:
        #
        # >>> import tz, datetime
        # >>> t = tz.tzlocal()
        # >>> datetime.datetime(2003,2,15,23,tzinfo=t).tzname()
        # 'BRDT'
        # >>> datetime.datetime(2003,2,16,0,tzinfo=t).tzname()
        # 'BRST'
        # >>> datetime.datetime(2003,2,15,23,tzinfo=t).tzname()
        # 'BRST'
        # >>> datetime.datetime(2003,2,15,22,tzinfo=t).tzname()
        # 'BRDT'
        # >>> datetime.datetime(2003,2,15,23,tzinfo=t).tzname()
        # 'BRDT'
        #
        # Here is a more stable implementation:
        #
        if not self._hasdst:
            return False

        # Check for ambiguous times:
        dstval = self._naive_is_dst(dt)

        if self.is_ambiguous(dt):
            return not dt.fold

        return dstval

    def __eq__(self, other):
        if isinstance(other, tzlocal):
            return (self._std_offset == other._std_offset and
                    self._dst_offset == other._dst_offset)
        elif isinstance(other, tzutc):
            return (not self._hasdst and
                    self._tznames[0] in {'UTC', 'GMT'} and
                    self._std_offset == ZERO)
        elif isinstance(other, tzoffset):
            return (not self._hasdst and
                    self._tznames[0] == other._name and
                    self._std_offset == other._offset)
        else:
            return NotImplemented

    __hash__ = None

    def __repr__(self):
        return "%s()" % self.__class__.__name__

    __reduce__ = object.__reduce__



def _datetime_to_timestamp(dt):
    """
    Convert a :class:`datetime.datetime` object to an epoch timestamp in
    seconds since January 1, 1970, ignoring the time zone.
    """
    return (dt.replace(tzinfo=None) - EPOCH).total_seconds()
