"""Token classes (``sqlparse.sql``): sqlparse's own classes when it is installed."""

from pyroid.sqlparse import _backend

if _backend.REAL is not None:
    _src = _backend.REAL.sql
else:
    from pyroid.sqlparse import _own_sql as _src

    for _name in ("NameAliasMixin", *_backend.CLASSES):
        getattr(_src, _name).__module__ = __name__

globals().update({k: v for k, v in vars(_src).items() if not k.startswith("__")})
del _src
