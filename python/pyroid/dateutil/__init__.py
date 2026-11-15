"""pyroid's drop-in for python-dateutil's date parser (``import pyroid.dateutil.parser``).

:mod:`pyroid.dateutil.parser` reproduces ``dateutil.parser``: ``parse``, ``parser``,
``parserinfo``, ``isoparse`` and ``isoparser``, with the installed python-dateutil's time zone
objects and exceptions. The rest of python-dateutil (``tz``, ``relativedelta``, ``rrule``,
``easter``) is not ported: use dateutil's own.
"""

from __future__ import annotations

from pyroid.dateutil import parser

__all__ = ["parser"]
