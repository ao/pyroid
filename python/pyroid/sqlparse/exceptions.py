"""``sqlparse.exceptions``: sqlparse's own ``SQLParseError`` when it is installed."""

from pyroid.sqlparse import _backend

if _backend.REAL is not None:
    _src = _backend.REAL.exceptions
else:
    from pyroid.sqlparse import _own_exceptions as _src

    _src.SQLParseError.__module__ = __name__

globals().update({k: v for k, v in vars(_src).items() if not k.startswith("__")})
del _src
