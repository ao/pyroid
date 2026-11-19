# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""langdetect's ``Language`` (langdetect's own class when it is installed)."""

from typing import Any

__all__ = ["Language"]

class Language:
    """A detected language: ``lang`` and ``prob``.

    ``repr`` is ``'lang:prob'``, and ``<`` orders by ``prob``.
    """

    lang: Any
    prob: Any
    def __init__(self, lang: Any, prob: Any) -> None: ...
    def __lt__(self, other: Any) -> Any: ...
