"""Pyroid: a Rust-fast text-cleaning core for LLM datasets.

Version 1.0 is a ground-up rewrite in progress. The 0.7.x API is gone; it
lives on the ``legacy`` branch and as the 0.7.x releases on PyPI.

Functions live in submodules (``import pyroid.text``). Each accepts a ``str``,
a ``list``/``tuple`` of ``str``, or an Arrow string array/column (anything with
``__arrow_c_stream__`` or ``__arrow_c_array__``: pyarrow, polars, pandas, HF
datasets, ...), and returns the same kind; Arrow input gives a
:class:`pyroid.Column`.
"""

# The native extension is imported unconditionally. If it is missing or broken,
# importing pyroid fails: there are no pure-Python fallbacks or dummy objects.
try:
    from pyroid import _pyroid
except ImportError as exc:
    raise ImportError(
        "pyroid's native extension (pyroid._pyroid) failed to load. Install a "
        "pyroid wheel built for this Python, or build one from a source "
        "checkout with `maturin develop`."
    ) from exc

from pyroid._pyroid import Column, InputError, PyroidError

__version__: str = _pyroid.__version__

__all__ = ["Column", "InputError", "PyroidError", "__version__"]
