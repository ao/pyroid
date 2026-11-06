"""Pyroid: a Rust-fast text-cleaning core for LLM datasets.

Version 1.0 is a ground-up rewrite in progress. The 0.7.x API is gone; it
lives on the ``legacy`` branch and as the 0.7.x releases on PyPI.

Functions live in submodules (``import pyroid.text``). Each accepts a ``str``,
a ``list``/``tuple`` of ``str``, or an Arrow string array/column (anything with
``__arrow_c_stream__`` or ``__arrow_c_array__``: pyarrow, polars, pandas, HF
datasets, ...), and returns the same kind; Arrow input gives a
:class:`pyroid.Column`.
"""

from types import ModuleType


def _load_extension() -> ModuleType:
    # The native extension is imported unconditionally. If it is missing or
    # broken, importing pyroid fails: no pure-Python fallbacks, no dummies.
    try:
        from pyroid import _pyroid
    except ImportError as exc:
        raise ImportError(
            "pyroid's native extension (pyroid._pyroid) failed to load. Install a "
            "pyroid wheel built for this Python, or build one from a source "
            "checkout with `maturin develop`."
        ) from exc
    return _pyroid


_pyroid = _load_extension()

from pyroid._pyroid import Column, FallbackWarning, InputError, PyroidError  # noqa: E402

__version__: str = _pyroid.__version__


def patch(*names: str) -> None:
    """Make ``import ftfy`` / ``import fasttext`` / ``import anyascii`` give pyroid's drop-ins.

    ``pyroid.patch("ftfy", "fasttext")`` is :func:`pyroid.integrations.patch`: explicit,
    reversible ``sys.modules`` shims (see there). Nothing is imported until it is called.
    """
    from pyroid.integrations._patch import patch as _patch

    _patch(*names)


def unpatch(*names: str) -> None:
    """Undo :func:`patch` for ``names`` (all if none): :func:`pyroid.integrations.unpatch`."""
    from pyroid.integrations._patch import unpatch as _unpatch

    _unpatch(*names)


__all__ = [
    "Column",
    "FallbackWarning",
    "InputError",
    "PyroidError",
    "__version__",
    "patch",
    "unpatch",
]
