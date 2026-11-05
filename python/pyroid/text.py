"""Whitespace normalisation and length filtering (``pyroid.text``).

The demonstration module for pyroid's conventions: a transform
(:func:`normalize_whitespace`) and a predicate (:func:`min_chars`). Types and
docs are in ``text.pyi``.
"""

from pyroid._pyroid import text as _native

normalize_whitespace = _native.normalize_whitespace
min_chars = _native.min_chars

__all__ = ["min_chars", "normalize_whitespace"]
