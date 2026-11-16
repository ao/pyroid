# Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
# contributors; modified.
"""``packaging.version``: PEP 440 versions (:class:`Version`, :func:`parse`)."""

from __future__ import annotations

from pyroid._pyroid import packaging as _native
from pyroid.packaging._compat import VERSION_PATTERN, InvalidVersion

__all__ = ["VERSION_PATTERN", "InvalidVersion", "Version", "normalize_pre", "parse"]

Version = _native.Version
parse = _native.parse
normalize_pre = _native.normalize_pre


def __dir__() -> list[str]:
    return __all__
