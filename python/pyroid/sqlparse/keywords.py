"""``sqlparse.keywords``: sqlparse's own when it is release 0.6.1, else pyroid's copy."""

from pyroid.sqlparse import _backend

if _backend.SHARE_LEXER and _backend.REAL is not None:
    _src = _backend.REAL.keywords
else:
    from pyroid.sqlparse import _own_keywords as _src

    _src.DelimitedSpans.__module__ = __name__

globals().update({k: v for k, v in vars(_src).items() if not k.startswith("__")})
del _src
