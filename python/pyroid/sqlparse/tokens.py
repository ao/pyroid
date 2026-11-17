"""Token types (``sqlparse.tokens``): sqlparse's own objects when it is installed."""

from pyroid.sqlparse import _backend

if _backend.REAL is not None:
    _src = _backend.REAL.tokens
else:
    from pyroid.sqlparse import _own_tokens as _src

    _src._TokenType.__module__ = __name__

globals().update({k: v for k, v in vars(_src).items() if not k.startswith("__")})
del _src
